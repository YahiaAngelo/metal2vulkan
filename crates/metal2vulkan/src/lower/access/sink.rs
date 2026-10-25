//! The stage's write-only sink for a dropped store through a buffer slot's descriptor view.
//!
//! A store through a slot bound to memory whose GPU writes Metal drops is dropped on the slot's
//! bit of the draw's or dispatch's `SpecKey::DroppedStoreSlots` word (`emit::writable`). The store
//! must not be guarded by a branch: a branch around it moves every value only that store consumes
//! into its block, and Apple's backend expands `air.fast_asinh.f32` and `air.fast_acosh.f32` to
//! different last bits depending on the basic block their call sits in, so a branching guard
//! changes what Metal's fast-math library answers.
//!
//! A slot reached by its device address needs nothing here: the bit joins the bounds check in the
//! one selection of the store's address, which sends the store to the root table's write-only sink
//! (`psb::prepare`, `memory::roots::SINK_ROOT`). A slot reached through a **descriptor view** has
//! no second buffer of its own storage class to select a pointer against, so the stage binds one:
//! [`SyntheticBinding::StoreSink`], a write-only storage buffer of
//! [`NeutralResource::SINK_BYTES`](crate::bind::NeutralResource::SINK_BYTES) bytes
//! (`WriteSource::Neutral(NeutralResource::Sink { .. })`), **one** storage descriptor of the stage
//! however many of its slots store. Every store through a view then runs unconditionally at
//! `select(in_bounds && keeps, own_place, sink_place)`, and the module branches for no store.
//!
//! The descriptor is [`reserve`]d as soon as a store through a buffer slot is selected, before the
//! slots' forms are chosen, so the per-stage descriptor fallbacks count it like any other view and
//! move slots to their device addresses (or read-only ones to uniform views) to make room for it;
//! [`finish`] gives it back where no slot kept a descriptor view in the end.
//!
//! The selection is a variable pointer: `OpSelect` between two `StorageBuffer` pointers, which
//! needs [`VkFeature::VariablePointers`] and the [`VkFeature::VariablePointersStorageBuffer`]
//! Vulkan requires beside it ([`selects_pointers`]; Vulkan 1.1, and every committed profile but
//! SwiftShader's and the Vulkan 1.2 floor has both). A stage of a device without it keeps the
//! branch, which moves no stored value there: the values the branch holds are computed by
//! instructions whose results do not depend on the block they sit in, and the one compiler that
//! answers otherwise is Apple's, on a device whose `float` functions are Metal's own
//! ([`FloatFunctions::Metal`], MoltenVK on an Apple GPU), which has the feature. A device that
//! both answers Metal's own functions and cannot select between two descriptors' pointers has no
//! exact form, so [`finish`] refuses it, naming the feature.
//!
//! The sink carries the same **unit views** as the slots whose dropped stores reach it (a `u8`, a
//! `u16` and a `uvec4` block at the one binding, `views::UnitViews`), because `OpSelect` needs both
//! pointers to have one result type: a dropped store of a narrow or 16-byte unit selects a pointer
//! of that unit's own type. They are further module-scope variables at the sink's one binding, not
//! further descriptors. The units are those of every access of such a slot, which is a superset of
//! what its stores move: a view the stores do not reach is a declaration nothing uses, where a
//! missing one would fail the build.
//!
//! Every dropped store lands at unit 0 of the sink, which is at least 64 bytes and so holds a
//! whole leaf; no shader reads it.
//!
//! The stage binds the sink whenever a gated store goes through a descriptor view, before the
//! leaves of those stores are known. A stage whose every such store merges a partial word, and so
//! keeps its branch (`leaf::merges`), therefore binds one descriptor it does not read. Deciding
//! otherwise would need the leaves here, where only the slots' forms and units are known; the
//! cost is one storage descriptor of such a stage.

use crate::device::profile::{DeviceProfile, FloatFunctions};
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::select::SelectCx;
use crate::memory::leaf::Units;
use crate::memory::views::{self, UnitViews, View, ViewKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;

use super::{AccessPlan, LowerFamily, SlotLowering};

/// What the stage's dropped stores through a descriptor view need of the sink: the unit views it
/// declares beside its word view, so each such store has a place of its own type to select
/// against (the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SinkPlan {
    pub units: Units,
}

/// The stage's declared sink: its word view, and the unit views beside it at the one binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SinkViews {
    pub view: View,
    pub units: UnitViews,
}

/// The binding symbol of `stage`'s store sink.
pub fn symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::StoreSink),
        index: 0,
    }
}

/// Whether the device can select between two descriptors' `StorageBuffer` pointers, which is what
/// a dropped store through a descriptor view lands in the sink by: `variablePointers`, and the
/// `variablePointersStorageBuffer` Vulkan requires beside it. The narrower feature alone is not
/// enough -- its capability holds only for arms pointing into one memory object ("Variable
/// pointers must point into the same structure") -- and no committed profile has one without the
/// other.
pub fn selects_pointers(profile: &DeviceProfile) -> bool {
    let enabled = &profile.fields().enabled.features;
    enabled.contains(&VkFeature::VariablePointers)
        && enabled.contains(&VkFeature::VariablePointersStorageBuffer)
}

/// Count the stage's sink descriptor, once, where the device can select pointers: called as the
/// first store through a buffer slot asks for the dropped-store word
/// (`super::request_dropped_store_word`), so the per-stage descriptor fallbacks see it before they
/// choose the slots' forms. [`finish`] gives it back where the forms left no descriptor view.
pub(super) fn reserve(cx: &mut SelectCx<'_>) {
    if !cx.families.access.dropped_store_slots.is_empty() || !selects_pointers(cx.profile) {
        return;
    }
    cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_add(1);
}

/// State the stage's sink where a dropped store through a descriptor view needs it, once the
/// slots' forms are final (`lower::select`), and give the [`reserve`]d descriptor back where none
/// does.
///
/// A device that cannot select pointers keeps the branch, which moves no stored value there (the
/// module docs); one that both keeps the branch and answers Metal's own `float` functions has no
/// exact form and is refused, naming the feature it lacks.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let access = &cx.families.access;
    let mut units = Units::NONE;
    let mut needed = false;
    for &i in &access.dropped_store_slots {
        if !matches!(
            access.slots.get(i).copied().flatten(),
            Some(SlotLowering::View(_))
        ) {
            continue;
        }
        needed = true;
        units = units.union(access.units.get(&i).copied().unwrap_or(Units::NONE));
    }
    let reserved = !access.dropped_store_slots.is_empty() && selects_pointers(cx.profile);
    if !needed {
        if reserved {
            cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_sub(1);
        }
        return Ok(());
    }
    if !reserved {
        if cx.profile.fields().float_functions == FloatFunctions::Metal {
            return Err(Unsupported::needs(
                Feature::Lowering(LowerFamily::Access),
                Capability::Feature(VkFeature::VariablePointers),
            )
            .into());
        }
        return Ok(());
    }
    cx.families.access.store_sink = Some(SinkPlan { units });
    Ok(())
}

/// Declare the stage's sink views where it binds one ([`finish`]): a storage word view at its
/// binding symbol, and the unit views the plan asks for beside it.
pub fn declare(
    mb: &mut ModuleBuilder,
    stage: Stage,
    plan: &AccessPlan,
) -> BResult<Option<SinkViews>> {
    let Some(p) = plan.store_sink else {
        return Ok(None);
    };
    let sym = symbol(stage);
    let view = views::declare_at(mb, sym, ViewKind::Storage)?;
    let units = UnitViews::declare_at(mb, sym, view, p.units)?;
    Ok(Some(SinkViews { view, units }))
}
