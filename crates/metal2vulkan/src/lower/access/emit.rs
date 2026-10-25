//! Emission of the access strategies.

use std::collections::BTreeSet;

use crate::air::ir::InstId;
use crate::lower::emit::EmitCx;
use crate::memory::contain::{self, Check};
use crate::memory::leaf::{self, k32, u32_op, Leaf, Units, WordMem, WordOps};
use crate::memory::views::{self, UnitViews, View, ViewKind};
use crate::pipeline::sym::{PushField, SpecKey};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value as SValue};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstKind, Constant, MemAccess, Scope, Semantics, Type, UnOp, ValueOrigin,
};

use super::psb::{self, PsbMem};
use super::{
    Access, AccessStrategy, Gep, IndexForm, PtrOp, PtrSide, SlotLowering, Target, BLOB_ROOT,
};

/// The word memory of one descriptor view, loaded and stored with memory operands (a coherence,
/// and `Volatile`; a uniform view's read-only words stay plain: no agent writes them while the
/// dispatch runs). Sub-word stores are two relaxed device-scope word atomics, except where the
/// slot's narrow views hold the leaf's units; a 16-byte aligned leaf is one `uvec4` where the slot
/// has that view.
#[derive(Clone, Copy, Debug)]
pub(super) struct ViewMem {
    view: View,
    ops: WordOps,
    units: UnitViews,
    /// The condition under which a store lands at the view's own place: `None` for a store that
    /// always lands there and for every load. Where it is given, every store selects its place
    /// between the view's and the stage's write-only sink, so no store branches
    /// ([`store_one`], `super::sink`).
    at_own: Option<SValue>,
    /// The stage's write-only sink, where it binds one (`super::sink::declare`).
    sink: Option<super::sink::SinkViews>,
}

impl ViewMem {
    /// The memory of `view`, with plain accesses and no unit view: the constant blob's
    /// ([`super::blob`]), which nothing stores to.
    pub(super) fn plain(view: View) -> ViewMem {
        ViewMem {
            view,
            ops: WordOps::default(),
            units: UnitViews::NONE,
            at_own: None,
            sink: None,
        }
    }

    /// The memory of slot view `view` with unit views `units` and the stage's sink `sink`.
    pub(super) fn of(
        view: View,
        units: UnitViews,
        sink: Option<super::sink::SinkViews>,
    ) -> ViewMem {
        ViewMem {
            view,
            ops: WordOps::default(),
            units,
            at_own: None,
            sink,
        }
    }

    /// The same memory, loaded and stored with `ops`.
    fn with_ops(self, ops: WordOps) -> ViewMem {
        ViewMem { ops, ..self }
    }

    /// The same memory, every store of it landing at its own place only where `at_own` holds and
    /// in the stage's sink otherwise. `None` where the form has no sink to select against, which
    /// leaves the condition to [`store_one`]'s branch.
    fn dropping(self, at_own: SValue) -> Option<ViewMem> {
        self.sink.map(|_| ViewMem {
            at_own: Some(at_own),
            ..self
        })
    }

    /// The slot's word view.
    pub(super) fn view(&self) -> View {
        self.view
    }

    /// The narrow view of `bytes`-byte units: present whenever [`WordMem::units`] says so.
    fn narrow(&self, bytes: u32) -> BResult<views::NarrowView> {
        self.units.of(bytes).ok_or(self.missing())
    }

    /// The `uvec4` view: present whenever [`WordMem::units`] says so.
    fn vec4(&self) -> BResult<views::Vec4View> {
        self.units.vec4.ok_or(self.missing())
    }

    fn missing(&self) -> crate::spirv::error::BuildError {
        crate::spirv::error::BuildError::Dangling(crate::spirv::error::Dangling::Global(
            self.view.global,
        ))
    }

    /// The place a store lands at: `own`, or, under [`ViewMem::at_own`]'s negation, the sink's
    /// place `sink_of` gives for the same unit, selected between the two pointers (one `OpSelect`
    /// of `StorageBuffer` pointers, `VariablePointersStorageBuffer`). Every dropped store lands at
    /// unit 0 of the sink, which holds a whole leaf.
    fn store_place(
        &self,
        fb: &mut FunctionBuilder<'_>,
        own: SValue,
        sink_of: impl FnOnce(
            &mut FunctionBuilder<'_>,
            super::sink::SinkViews,
            SValue,
        ) -> BResult<SValue>,
    ) -> BResult<SValue> {
        let Some(at_own) = self.at_own else {
            return Ok(own);
        };
        let sink = self.sink.ok_or(self.missing())?;
        let zero = k32(fb, 0)?;
        let at_sink = sink_of(fb, sink, zero)?;
        fb.select(at_own, own, at_sink)
    }
}

impl WordMem for ViewMem {
    fn read_vec4(&mut self, fb: &mut FunctionBuilder<'_>, index: SValue) -> BResult<SValue> {
        let view = self.vec4()?;
        // A uniform view's read-only vectors are read plainly, as its words are.
        let access = match view.class {
            crate::spirv::ir::StorageClass::Uniform => MemAccess::default(),
            _ => self.ops.access(fb, MemAccess::default(), false)?,
        };
        views::read_vec4(fb, view, index, access)
    }

    fn write_vec4(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        index: SValue,
        value: SValue,
    ) -> BResult<()> {
        let view = self.vec4()?;
        let access = self.ops.access(fb, MemAccess::default(), true)?;
        let own = views::vec4_ptr(fb, view, index)?;
        let p = self.store_place(fb, own, |fb, sink, zero| {
            let view = sink.units.vec4.ok_or(missing(sink.view))?;
            views::vec4_ptr(fb, view, zero)
        })?;
        fb.store(p, value, access)
    }

    fn units(&self) -> Units {
        self.units.units()
    }

    fn read_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: SValue,
    ) -> BResult<SValue> {
        let view = self.narrow(bytes)?;
        let access = self.ops.access(fb, MemAccess::default(), false)?;
        views::read_unit(fb, view, index, access)
    }

    fn write_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: SValue,
        value: SValue,
    ) -> BResult<()> {
        let view = self.narrow(bytes)?;
        let access = self.ops.access(fb, MemAccess::default(), true)?;
        let own = views::unit_ptr(fb, view, index)?;
        let p = self.store_place(fb, own, |fb, sink, zero| {
            let view = sink.units.of(bytes).ok_or(missing(sink.view))?;
            views::unit_ptr(fb, view, zero)
        })?;
        views::write_unit(fb, view, p, value, access)
    }

    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        if self.ops == WordOps::default() || self.view.kind != ViewKind::Storage {
            return views::read_word(fb, self.view, word);
        }
        let p = storage_ptr(fb, self.view, word)?;
        let access = self.ops.access(fb, MemAccess::default(), false)?;
        fb.load(p, access)
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue, value: SValue) -> BResult<()> {
        let p = self.store_word_place(fb, word)?;
        let access = self.ops.access(fb, MemAccess::default(), true)?;
        fb.store(p, value, access)
    }

    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: SValue,
        mask: SValue,
        bits: SValue,
    ) -> BResult<()> {
        let p = self.store_word_place(fb, word)?;
        let keep = fb.unary(UnOp::Not, mask)?;
        let clear = fb.binary(BinOp::BitwiseOr, keep, bits)?;
        fb.atomic(AtomicOp::And, p, Scope::Device, Semantics::RELAXED, clear)?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bits)?;
        Ok(())
    }
}

/// A missing view of `view`'s binding: an internal invariant failure (the stage's sink declares
/// the unit views of every slot whose dropped stores reach it, `super::sink`).
fn missing(view: View) -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Dangling(crate::spirv::error::Dangling::Global(view.global))
}

impl ViewMem {
    /// The place a word store lands at ([`ViewMem::store_place`] over the word views).
    fn store_word_place(&self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let own = storage_ptr(fb, self.view, word)?;
        self.store_place(fb, own, |fb, sink, zero| storage_ptr(fb, sink.view, zero))
    }
}

/// A word pointer of a storage view. Stores through a uniform view are refused by selection, so a
/// missing pointer is an internal invariant failure.
fn storage_ptr(fb: &mut FunctionBuilder<'_>, view: View, word: SValue) -> BResult<SValue> {
    match views::word_ptr(fb, view, word)? {
        Some(p) => Ok(p),
        None => Err(crate::spirv::error::BuildError::Dangling(
            crate::spirv::error::Dangling::Global(view.global),
        )),
    }
}

/// The word memory of one target: a descriptor view or a device address.
#[derive(Clone, Copy, Debug)]
pub(super) enum Mem {
    View(ViewMem),
    Psb(PsbMem),
}

impl Mem {
    /// The same memory, loaded and stored with `ops`.
    pub(super) fn with(self, ops: WordOps) -> Mem {
        match self {
            Mem::View(m) => Mem::View(ViewMem { ops, ..m }),
            Mem::Psb(m) => Mem::Psb(m.with(ops)),
        }
    }
}

impl WordMem for Mem {
    fn read(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        match self {
            Mem::View(m) => m.read(fb, word),
            Mem::Psb(m) => m.read(fb, word),
        }
    }

    fn write(&mut self, fb: &mut FunctionBuilder<'_>, word: SValue, value: SValue) -> BResult<()> {
        match self {
            Mem::View(m) => m.write(fb, word, value),
            Mem::Psb(m) => m.write(fb, word, value),
        }
    }

    fn write_bits(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        word: SValue,
        mask: SValue,
        bits: SValue,
    ) -> BResult<()> {
        match self {
            Mem::View(m) => m.write_bits(fb, word, mask, bits),
            Mem::Psb(m) => m.write_bits(fb, word, mask, bits),
        }
    }

    fn units(&self) -> Units {
        match self {
            Mem::View(m) => m.units(),
            Mem::Psb(m) => m.units(),
        }
    }

    fn read_vec4(&mut self, fb: &mut FunctionBuilder<'_>, index: SValue) -> BResult<SValue> {
        match self {
            Mem::View(m) => m.read_vec4(fb, index),
            Mem::Psb(m) => m.read_vec4(fb, index),
        }
    }

    fn write_vec4(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        index: SValue,
        value: SValue,
    ) -> BResult<()> {
        match self {
            Mem::View(m) => m.write_vec4(fb, index, value),
            Mem::Psb(m) => m.write_vec4(fb, index, value),
        }
    }

    fn read_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: SValue,
    ) -> BResult<SValue> {
        match self {
            Mem::View(m) => m.read_unit(fb, bytes, index),
            Mem::Psb(m) => m.read_unit(fb, bytes, index),
        }
    }

    fn write_unit(
        &mut self,
        fb: &mut FunctionBuilder<'_>,
        bytes: u32,
        index: SValue,
        value: SValue,
    ) -> BResult<()> {
        match self {
            Mem::View(m) => m.write_unit(fb, bytes, index, value),
            Mem::Psb(m) => m.write_unit(fb, bytes, index, value),
        }
    }
}

/// One candidate slot, ready to access: its word memory, the byte offset in it, and the
/// in-shader containment condition when it has one.
#[derive(Clone, Copy, Debug)]
pub(super) struct Prepared {
    pub(super) mem: Mem,
    pub(super) off: SValue,
    pub(super) inb: Option<SValue>,
    /// A store needs `if_(inb)`: a view's out-of-bounds offset is not redirected (a PSB access
    /// already points at the sink).
    pub(super) guard_store: bool,
    /// The draw's or dispatch's word says the slot's memory keeps the GPU's writes, so a store
    /// through memory whose writes Metal drops is dropped ([`writable`],
    /// `SpecKey::DroppedStoreSlots`). `None` where the access stores nothing, at a slot no guest
    /// buffer can be bound to, and on the device-address form, which has already selected the
    /// sink's address and so needs no condition here (`psb::prepare`).
    pub(super) writable: Option<SValue>,
}

/// Whether the draw's or dispatch's `SpecKey::DroppedStoreSlots` word leaves `slot`'s stores
/// standing: `(word >> slot.index) & 1 == 0`, one uniform word read and three uniform
/// instructions, which every store through the slot reads. `None` at a slot that is not a
/// guest buffer slot of the stage's argument table (a synthetic binding, which the runtime owns
/// and never places over a caller's read-only pages) and so has no bit of the word.
///
/// The condition selects where the store lands wherever the form has a sink to select: the
/// device-address form puts it beside the bounds check in the one address selection
/// (`psb::prepare`), so no store branches on it. A descriptor view has no second buffer of its own
/// storage class to select against, so there the store still sits in `if_`, which moves what
/// Apple's fast-math instructions answer (`store_one`).
fn writable(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    slot: crate::pipeline::sym::SlotRef,
) -> BResult<Option<SValue>> {
    if slot.class != crate::pipeline::sym::SlotClass::Buffer || slot.index >= u32::BITS {
        return Ok(None);
    }
    let field = PushField::Specialized(SpecKey::DroppedStoreSlots(slot.stage));
    let word = cx.push_field(fb, field)?;
    let bit = k32(fb, 1 << slot.index)?;
    let dropped = u32_op(fb, BinOp::BitwiseAnd, word, bit)?;
    let zero = k32(fb, 0)?;
    Ok(Some(fb.binary(BinOp::IEqual, dropped, zero)?))
}

/// Prepare `t` for an access of `n` bytes at Metal offset `m`, which stores where `store`.
pub(super) fn prepare(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Target,
    m: SValue,
    n: u32,
    store: bool,
) -> BResult<Prepared> {
    let Some(choice) = cx.plan.slots.get(t.slot).copied().flatten() else {
        return Err(cx.fail("an access names an unplanned slot"));
    };
    let writable = match store {
        true => writable(cx, fb, choice.slot)?,
        false => None,
    };
    let lowering = cx.plan.families.access.slots.get(t.slot).copied().flatten();
    let ops = WordOps::of(crate::lower::sync::memory_model::coherence(cx.plan));
    if let Some(SlotLowering::Psb { start }) = lowering {
        let table = cx.push_field(fb, PushField::RootTableAddress)?;
        let contained = cx.plan.families.access.psb_contained;
        let a = psb::prepare(fb, table, t.root, m, start, n, contained, writable)?;
        return Ok(Prepared {
            mem: Mem::Psb(a.mem.with(ops)),
            off: a.off,
            inb: a.inb,
            guard_store: false,
            writable: None,
        });
    }
    let Some(view) = cx.decl.families.access.views.get(t.slot).copied().flatten() else {
        return Err(cx.fail("an access names an undeclared slot view"));
    };
    let residual = cx.push_field(fb, PushField::SlotBindOffset(choice.slot))?;
    let shifted = u32_op(fb, BinOp::IAdd, residual, m)?;
    // Where the binder starts a uniform view's range at the bind offset (`RangeAtBindOffset`), the
    // residual is 0: the offset alone, a constant for a constant member, which a driver can push.
    let off = match cx
        .decl
        .families
        .access
        .at_bind
        .get(t.slot)
        .copied()
        .flatten()
    {
        Some(at_bind) => fb.select(at_bind, m, shifted)?,
        None => shifted,
    };
    let inb = match (t.check, view.kind) {
        (Check::None, _) => None,
        (Check::InShader, ViewKind::Storage) => match views::storage_bytes(fb, view)? {
            Some(size) => Some(contain::in_bounds(fb, off, size, u64::from(n))?),
            None => return Err(cx.fail("a storage view without a size")),
        },
        (Check::InShader, ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. }) => {
            // Selection gives a uniform view only to an encoding without checks: the range bound
            // behind it may be shorter than its declared size.
            return Err(cx.fail("an in-shader check of a uniform view"));
        }
    };
    let (off, guard_store) = match inb {
        // Read at a clamped offset (word 0 exists: a bound range is never empty).
        Some(inb) => {
            let zero = k32(fb, 0)?;
            (fb.select(inb, off, zero)?, true)
        }
        None => (off, false),
    };
    let units = cx
        .decl
        .families
        .access
        .units
        .get(t.slot)
        .copied()
        .unwrap_or(UnitViews::NONE);
    let sink = cx.decl.families.access.store_sink;
    Ok(Prepared {
        mem: Mem::View(ViewMem::of(view, units, sink).with_ops(ops)),
        off,
        inb,
        guard_store,
        writable,
    })
}

/// Load `leaf` from `p`, contained: the value, or 0 out of bounds.
fn load_one(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, p: Prepared) -> BResult<SValue> {
    let mut mem = p.mem;
    let v = leaf::load(fb, &mut mem, leaf, p.off)?;
    match p.inb {
        None => Ok(v),
        Some(inb) => {
            let z = leaf::zero(fb, leaf)?;
            fb.select(inb, v, z)
        }
    }
}

/// Store `v` as `leaf` at `p`, contained and only where the slot's memory keeps the GPU's writes
/// ([`Prepared::writable`]). Returns whether the store merges a partial word ([`leaf::store`]).
///
/// A condition left here is one `prepare` could not fold into an address selection: the
/// device-address form leaves none ([`Prepared::writable`], `psb::prepare`). Where the stage binds
/// its write-only sink, the store runs unconditionally at a place selected between the view's own
/// and the sink's (`super::sink`); only where it binds none does the store sit in `if_`.
///
/// That branch is not free of meaning: a value a guarded store alone consumes is written inside
/// the branch, and Apple's backend expands `air.fast_asinh.f32` and `air.fast_acosh.f32` to
/// different last bits depending on the basic block their call sits in, so the module records
/// `PrecisionNote::GuardedStoreBranch` where it keeps one.
fn store_one(fb: &mut FunctionBuilder<'_>, leaf: &Leaf, p: Prepared, v: SValue) -> BResult<bool> {
    let mut mem = p.mem;
    let contained = p.guard_store.then_some(p.inb).flatten();
    let guard = match (contained, p.writable) {
        (Some(inb), Some(keeps)) => Some(fb.binary(BinOp::LogicalAnd, inb, keeps)?),
        (Some(inb), None) => Some(inb),
        (None, Some(keeps)) => Some(keeps),
        (None, None) => None,
    };
    let Some(guard) = guard else {
        return leaf::store(fb, &mut mem, leaf, p.off, v);
    };
    // The selecting form: no branch, and a dropped or uncontained store lands in the sink, which
    // no shader reads. The offset is already clamped for containment, so the view's own place is
    // inside its range whatever the condition says. A store that merges a partial word keeps the
    // branch: its two word atomics need a place, and an `OpSelect` of two pointers is not one
    // (SPIRV-Cross writes an atomic's pointer as `&expr`, an lvalue in MSL), so such a store is
    // still a gap (docs/FAST-PATHS.md, known gaps).
    if let Mem::View(view) = mem {
        if !leaf::merges(leaf, view.units()) {
            if let Some(view) = view.dropping(guard) {
                let mut mem = Mem::View(view);
                return leaf::store(fb, &mut mem, leaf, p.off, v);
            }
        }
    }
    let leaf = *leaf;
    let then: ArmFn<'_, bool> = Box::new(move |fb, merged| {
        let mut mem = p.mem;
        *merged = leaf::store(fb, &mut mem, &leaf, p.off, v)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    let mut merged = false;
    fb.if_(&mut merged, guard, then, None)?;
    Ok(merged)
}

/// Part `i` of fat pointer `p` (0 the root, 1 the offset): a constant pointer's part, known at
/// translation time, or the extracted word.
pub(super) fn fat_part(fb: &mut FunctionBuilder<'_>, p: SValue, i: u32) -> BResult<SValue> {
    if let ValueOrigin::Const(c) = fb.origin(p)? {
        let part = match &fb.module().consts().get(c)?.kind {
            ConstKind::Plain(Constant::Composite { parts, .. }) if parts.len() == 2 => {
                parts.get(i as usize).copied()
            }
            _ => None,
        };
        if let Some(part) = part {
            return fb.module().const_value(part);
        }
    }
    fb.composite_extract(p, &[i])
}

/// The fat pointer `(root, off)` of type `ty`: a constant when both parts are known.
fn fat(fb: &mut FunctionBuilder<'_>, ty: TypeId, root: SValue, off: SValue) -> BResult<SValue> {
    if let (Some(r), Some(o)) = (leaf::known(fb, root)?, leaf::known(fb, off)?) {
        let r = fb.u32_const(r)?;
        let o = fb.u32_const(o)?;
        let c = fb.module().constant(Constant::Composite {
            ty,
            parts: [r, o].into(),
        })?;
        return fb.module().const_value(c);
    }
    fb.composite_construct(ty, &[root, off])
}

/// The accessed value of a load, or the stored value's store.
fn access(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &Access,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let p = cx.operand(fb, a.ptr)?;
    let m = fat_part(fb, p, 1)?;
    access_at(cx, fb, a, p, m, stored)
}

/// The access `a` of its leaf at Metal offset `m` of fat pointer `p` (`a.ptr`'s value, or that
/// pointer displaced by a memory intrinsic): the loaded value, or the stored value's store.
pub(super) fn access_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &Access,
    p: SValue,
    m: SValue,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    // A `*_coherent` access's own coherence, or the function's; `Volatile` for a volatile one.
    let ops = WordOps {
        coherence: a
            .coherence
            .unwrap_or_else(|| crate::lower::sync::memory_model::coherence(cx.plan)),
        volatile: a.volatile,
    };
    let mut prepared = Vec::with_capacity(a.targets.len());
    for t in &a.targets {
        let mut p = prepare(cx, fb, t, m, a.leaf.bytes, stored.is_some())?;
        p.mem = p.mem.with(ops);
        prepared.push((t.root, t.slot, p));
    }
    let leaf = a.leaf;
    // A pointer that may point into the blob reads it under `BLOB_ROOT` (never stored).
    let blob = match a.blob {
        Some(check) => match cx.decl.families.access.blob {
            Some(view) => Some((view, check == Check::InShader)),
            None => return Err(cx.fail("the constant blob is not declared")),
        },
        None => None,
    };
    if let Some((view, checked)) = blob {
        if stored.is_some() {
            return Err(cx.fail("a store through a pointer into the constant blob"));
        }
        let root = fat_part(fb, p, 0)?;
        let mut cases: Vec<(Vec<u32>, ArmFn<'_, ()>)> = Vec::with_capacity(prepared.len() + 1);
        for (r, _, prep) in prepared.iter().copied() {
            let arm: ArmFn<'_, ()> =
                Box::new(move |fb, _| Ok(ArmEnd::FallThrough(vec![load_one(fb, &leaf, prep)?])));
            cases.push((vec![r], arm));
        }
        let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
            Ok(ArmEnd::FallThrough(vec![super::blob::load_view(
                fb, view, &leaf, m, checked,
            )?]))
        });
        cases.push((vec![BLOB_ROOT], arm));
        let default: ArmFn<'_, ()> =
            Box::new(move |fb, _| Ok(ArmEnd::FallThrough(vec![leaf::zero(fb, &leaf)?])));
        let merged = fb.switch_at_use(&mut (), root, cases, default)?;
        return Ok(merged.0.first().copied());
    }
    match (prepared.as_slice(), stored) {
        // No candidate slot: only null, whose dereference Metal leaves undefined.
        ([], None) => Ok(Some(leaf::zero(fb, &leaf)?)),
        ([], Some(_)) => Ok(None),
        // One candidate: a null root dereferences it at the null pointer's offset, which Metal
        // leaves undefined and containment keeps inside the slot.
        ([(_, _, one)], None) => Ok(Some(load_one(fb, &leaf, *one)?)),
        ([(_, slot, one)], Some(v)) => {
            if store_one(fb, &leaf, *one, v)? {
                cx.merged.insert(*slot);
            }
            Ok(None)
        }
        (many, stored) => {
            let root = fat_part(fb, p, 0)?;
            // The slots whose arm's store merged a partial word.
            type Slots = BTreeSet<usize>;
            let mut cases: Vec<(Vec<u32>, ArmFn<'_, Slots>)> = Vec::with_capacity(many.len());
            for (r, slot, prep) in many.iter().copied() {
                let arm: ArmFn<'_, Slots> = Box::new(move |fb, merged| match stored {
                    None => Ok(ArmEnd::FallThrough(vec![load_one(fb, &leaf, prep)?])),
                    Some(v) => {
                        if store_one(fb, &leaf, prep, v)? {
                            merged.insert(slot);
                        }
                        Ok(ArmEnd::FallThrough(Vec::new()))
                    }
                });
                cases.push((vec![r], arm));
            }
            let default: ArmFn<'_, Slots> = Box::new(move |fb, _| match stored {
                None => Ok(ArmEnd::FallThrough(vec![leaf::zero(fb, &leaf)?])),
                Some(_) => Ok(ArmEnd::FallThrough(Vec::new())),
            });
            let mut merged = Slots::new();
            let out = fb.switch_at_use(&mut merged, root, cases, default)?;
            cx.merged.extend(merged);
            Ok(out.0.first().copied())
        }
    }
}

/// The index operand `x` (held in `form`) as the index value mod 2^32.
pub(super) fn index_u32(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    form: IndexForm,
) -> BResult<SValue> {
    match form {
        IndexForm::Word | IndexForm::WordZero => Ok(x),
        IndexForm::Narrow { bits } => {
            let s = k32(fb, 32 - u32::from(bits))?;
            let up = fb.binary(BinOp::ShiftLeftLogical, x, s)?;
            fb.binary(BinOp::ShiftRightArithmetic, up, s)
        }
        IndexForm::Bool => {
            let ones = k32(fb, u32::MAX)?;
            let zero = k32(fb, 0)?;
            fb.select(x, ones, zero)
        }
        IndexForm::BoolZero => {
            let one = k32(fb, 1)?;
            let zero = k32(fb, 0)?;
            fb.select(x, one, zero)
        }
        // A `u64`, or two words without `shaderInt64`.
        IndexForm::Wide => crate::lower::arith::wide::low_word(fb, x),
    }
}

fn gep(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, g: &Gep) -> BResult<SValue> {
    let p = cx.operand(fb, g.base)?;
    let root = fat_part(fb, p, 0)?;
    let mut off = fat_part(fb, p, 1)?;
    let c = k32(fb, g.konst)?;
    off = u32_op(fb, BinOp::IAdd, off, c)?;
    for (v, stride, form) in &g.terms {
        let x = cx.operand(fb, *v)?;
        let x = index_u32(fb, x, *form)?;
        let s = k32(fb, *stride)?;
        let x = u32_op(fb, BinOp::IMul, x, s)?;
        off = u32_op(fb, BinOp::IAdd, off, x)?;
    }
    let ty = fb.value_ty(p)?;
    fat(fb, ty, root, off)
}

/// The fat pointer of a `select` operand: a blob offset becomes `(BLOB_ROOT, offset)`, and the null
/// constant `(0, 0)`.
pub(super) fn side(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: PtrSide,
) -> BResult<SValue> {
    match s {
        PtrSide::Fat(v) => cx.operand(fb, v),
        PtrSide::Null => {
            let u = fb.module().ty(Type::Int {
                width: 32,
                signed: false,
            })?;
            let v2 = fb.module().ty(Type::Vector { elem: u, len: 2 })?;
            let zero = k32(fb, 0)?;
            fat(fb, v2, zero, zero)
        }
        PtrSide::Blob(p) => {
            let off = super::blob::ptr_value(cx, fb, p)?;
            let root = k32(fb, BLOB_ROOT)?;
            let u = fb.module().ty(Type::Int {
                width: 32,
                signed: false,
            })?;
            let v2 = fb.module().ty(Type::Vector { elem: u, len: 2 })?;
            fat(fb, v2, root, off)
        }
    }
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &AccessStrategy,
) -> BResult<()> {
    let Some(vid) = cx.plan.values.inst(at) else {
        return Err(cx.fail(format!("{at:?} is out of range")));
    };
    match s {
        AccessStrategy::Pointer(op) => {
            let r = match op {
                PtrOp::Gep(g) => gep(cx, fb, g)?,
                PtrOp::Copy { of } => cx.operand(fb, *of)?,
                PtrOp::Select { c, t, f } => {
                    let c = cx.operand(fb, *c)?;
                    let t = side(cx, fb, *t)?;
                    let f = side(cx, fb, *f)?;
                    fb.select(c, t, f)?
                }
            };
            cx.define(vid, r);
            Ok(())
        }
        AccessStrategy::Load(a) => match access(cx, fb, a, None)? {
            Some(v) => {
                cx.define(vid, v);
                Ok(())
            }
            None => Err(cx.fail("a load without a value")),
        },
        AccessStrategy::Store(a) => {
            let Some(val) = a.value else {
                return Err(cx.fail("a store without a value"));
            };
            let v = cx.operand(fb, val)?;
            access(cx, fb, a, Some(v)).map(|_| ())
        }
        AccessStrategy::Local(l) => super::local::emit(cx, fb, vid, l),
        AccessStrategy::Atomic(a) => super::atomic::emit(cx, fb, vid, a),
        AccessStrategy::Blob(b) => super::blob::emit(cx, fb, vid, b),
        AccessStrategy::Memory(m) => super::copy::emit(cx, fb, vid, m),
        AccessStrategy::Table(t) => super::buffer_table::emit(cx, fb, vid, t),
        AccessStrategy::Dynamic(d) => super::dynamic::emit(cx, fb, vid, d),
        AccessStrategy::Handle(h) => super::handle::emit(cx, fb, vid, h),
        AccessStrategy::Value(p) => super::pointer::emit(cx, fb, vid, p),
        AccessStrategy::Cell(c) => super::cell::emit(cx, fb, vid, c),
        AccessStrategy::Imageblock(i) => super::imageblock::emit(cx, fb, vid, i),
        AccessStrategy::Encode(e) => super::encode::emit(cx, fb, vid, e),
        AccessStrategy::FunctionTable(t) => super::function_table::emit(cx, fb, vid, t),
        AccessStrategy::AbsentRead(a) => super::absent::emit(cx, fb, vid, a),
    }
}
