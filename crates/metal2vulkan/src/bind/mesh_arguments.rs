//! The arguments pass of an indirect mesh draw on a device with `VK_EXT_mesh_shader`.
//!
//! `drawMeshThreadgroupsWithIndirectBuffer:` reads its group counts (an
//! `MTLDispatchThreadgroupsIndirectArguments`) as it runs. `vkCmdDrawMeshTasksIndirectEXT` requires
//! every count within the device's task or mesh limits, so the draw cannot read Metal's arguments
//! in place: the consumer records [`MeshArguments`] first, one invocation of the crate-owned
//! module [`mesh_arguments_module`] that reads Metal's three words and writes the
//! `VkDrawMeshTasksIndirectCommandEXT` the draw reads into a consumer-provided scratch of
//! [`MeshArguments::SCRATCH_BYTES`].
//!
//! **What Metal draws**, measured by our probe on an M3 Max (macOS 27.0.1), direct and indirect
//! draws alike, with and without API validation, which stops none of them:
//!
//! - Without an object function the counts are the mesh grid. A grid with an axis past
//!   [`METAL_MESH_GRID_AXIS`] (65,535) draws nothing; so does one of exactly 2^20 threadgroups
//!   (1024 by 1024, 128 by 128 by 64, 4096 by 256) and every one measured of 2^21 or more. Grids
//!   of up to [`METAL_MESH_GRID_TOTAL`] (1,048,575, Apple's limit for the family) draw whole. Past
//!   that Metal is erratic (17 by 61,681 and 65,535 by 24 drew, 1025 by 1025 and 33 by 32,768 did
//!   not), and the module draws nothing, Metal's answer at 2^20 and above 2^21.
//! - With an object function the counts are the object grid, which Metal draws at every count
//!   measured (16,777,216 along `x`). The mesh grid an object threadgroup sets follows the rule
//!   above (65,536 along `x` or `y`, 1024 by 1024 and 2048 by 1024 launch nothing from it; 1023 by
//!   1025 and 65,535 by 16 launch whole), which the object function's lowering holds it to
//!   (`lower::mesh`).
//!
//! **The module's rule** ([`MeshGridRule`], its constants): where Metal draws nothing, it writes
//! zero counts, which draw nothing. Where Metal draws and the counts lie within the device's
//! limits for the first stage (`maxMeshWorkGroupCount` and `maxMeshWorkGroupTotalCount`, or the
//! task stage's), it copies them. Where Metal draws and they do not (an object grid past the task
//! stage's limits), it writes zero counts and sets `MESH_GRID_PAST_LIMITS` in the command buffer's
//! `ExecutionStatus`, which the consumer reports as a command-buffer error: Metal draws them.
//! Every committed profile with the extension holds Metal's mesh grids within its mesh limits.
//!
//! **The module** (fixed per device and form): one `GLCompute` entry `main` with local size
//! `(1, 1, 1)`, no loops, three storage buffers in set 0 and one push constant:
//!
//! | Binding | Buffer | Range |
//! |---|---|---|
//! | [`MeshArgumentsModule::SOURCE_BINDING`] | the Metal indirect buffer | [`MeshArguments::source_range`] |
//! | [`MeshArgumentsModule::SCRATCH_BINDING`] | the consumer's scratch | [`MeshArguments::SCRATCH_BYTES`] |
//! | [`MeshArgumentsModule::STATUS_BINDING`] | the `ExecutionStatus` buffer | `ExecutionStatus::BYTES` |
//!
//! The push constant ([`MeshArguments::push_constants`], `COMPUTE`, offset 0) is the byte distance
//! from the bound source range's start to the arguments (a multiple of 4), since a storage
//! descriptor's offset must be a multiple of `minStorageBufferOffsetAlignment` where Metal's
//! argument offset is only 4-aligned. Reads of the source are contained (an index past the bound
//! range reads 0).
//!
//! **Synchronization** (consumer): the source write → `SHADER_READ` in `COMPUTE_SHADER` before
//! the pass; the scratch `SHADER_WRITE` → `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` between the
//! pass and the draw.

use std::sync::Arc;

use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage, Symbol, SyntheticBinding,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, LayoutRule,
    MemAccess, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// The most mesh threadgroups along an axis of a grid Metal draws without an object function
/// (the module docs).
pub const METAL_MESH_GRID_AXIS: u32 = 65_535;

/// The most mesh threadgroups of a grid Metal draws whole without an object function (the module
/// docs).
pub const METAL_MESH_GRID_TOTAL: u32 = 1_048_575;

/// Whether Metal draws nothing for a mesh grid of `groups` without an object function: an axis
/// past [`METAL_MESH_GRID_AXIS`], or more than [`METAL_MESH_GRID_TOTAL`] threadgroups.
pub(crate) fn metal_mesh_grid_draws_nothing(groups: [u32; 3]) -> bool {
    let total: u128 = groups.iter().map(|&g| u128::from(g)).product();
    groups.iter().any(|&g| g > METAL_MESH_GRID_AXIS) || total > u128::from(METAL_MESH_GRID_TOTAL)
}

/// The first stage of an indirect mesh draw, whose threadgroups Metal's arguments count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MeshArgumentsForm {
    /// The mesh function's, of a pipeline without an object function.
    Mesh,
    /// The object function's.
    Object,
}

/// A grid's limits: the largest count along each axis, and the largest product.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshGridLimits {
    /// The largest count along `x`, `y` and `z`.
    pub axes: [u32; 3],
    /// The largest product of the three counts.
    pub total: u32,
}

/// What [`mesh_arguments_module`] of one form holds the counts to (the module docs): Metal's
/// limits, past which it draws nothing, and the device's for the draw's first stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshGridRule {
    /// Metal's limits, or `None` where Metal draws every grid (an object grid).
    pub metal: Option<MeshGridLimits>,
    /// The device's limits for the first stage: mesh or task.
    pub device: MeshGridLimits,
}

impl MeshGridRule {
    /// The rule of `form` on a device of `limits`.
    pub fn of(form: MeshArgumentsForm, limits: &crate::device::MeshLimits) -> MeshGridRule {
        match form {
            MeshArgumentsForm::Mesh => MeshGridRule {
                metal: Some(MeshGridLimits {
                    axes: [METAL_MESH_GRID_AXIS; 3],
                    total: METAL_MESH_GRID_TOTAL,
                }),
                device: MeshGridLimits {
                    axes: limits.max_mesh_work_group_count,
                    total: limits.max_mesh_work_group_total_count,
                },
            },
            MeshArgumentsForm::Object => MeshGridRule {
                metal: None,
                device: MeshGridLimits {
                    axes: limits.max_task_work_group_count,
                    total: limits.max_task_work_group_total_count,
                },
            },
        }
    }
}

/// An arguments pass the consumer records before an indirect mesh draw on a device with
/// `VK_EXT_mesh_shader`: one invocation of [`mesh_arguments_module`] of the draw's form reads
/// Metal's arguments at `source` and writes the draw's `VkDrawMeshTasksIndirectCommandEXT` into a
/// consumer-provided scratch of [`MeshArguments::SCRATCH_BYTES`], from which the draw
/// (`DrawCommand::DrawMeshTasksIndirect` at offset 0) reads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshArguments {
    pub source: IndirectRef,
    /// The module of the draw's first stage.
    pub form: MeshArgumentsForm,
    /// The range of the source buffer to bind at [`MeshArgumentsModule::SOURCE_BINDING`]: from
    /// `source.offset` aligned down to `minStorageBufferOffsetAlignment` (and to 4) through the
    /// 12 argument bytes.
    pub source_range: BufferRange,
    /// The push constant bytes (`COMPUTE`, offset 0): the arguments' byte offset inside
    /// `source_range`.
    pub push_constants: [u8; 4],
}

impl MeshArguments {
    /// Bytes of the scratch the draw's arguments are written to
    /// (`VkDrawMeshTasksIndirectCommandEXT`).
    pub const SCRATCH_BYTES: u32 = 12;

    /// The pass for Metal arguments at `source` of a draw whose first stage is `form`, with the
    /// source bound at an offset aligned to `alignment` (`minStorageBufferOffsetAlignment`).
    /// `None` when `source.offset` is not a multiple of 4 (such a draw draws nothing) or the range
    /// overflows.
    pub fn new(source: IndirectRef, form: MeshArgumentsForm, alignment: u64) -> Option<Self> {
        if !source.offset.is_multiple_of(4) {
            return None;
        }
        let a = alignment.max(4);
        let start = source.offset - source.offset % a;
        let residual = u32::try_from(source.offset - start).ok()?;
        let end = source.offset.checked_add(u64::from(Self::SCRATCH_BYTES))?;
        Some(MeshArguments {
            source,
            form,
            source_range: BufferRange {
                offset: start,
                range: end - start,
            },
            push_constants: residual.to_le_bytes(),
        })
    }
}

/// The crate-owned arguments module of one device and form, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshArgumentsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
    /// The rule its constants hold.
    pub rule: MeshGridRule,
}

impl MeshArgumentsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The Metal indirect buffer, a storage buffer.
    pub const SOURCE_BINDING: u32 = 0;
    /// The consumer's scratch (`VkDrawMeshTasksIndirectCommandEXT`), a storage buffer.
    pub const SCRATCH_BINDING: u32 = 1;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 2;
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 4;
}

/// The binding symbol of the source buffer ([`MeshArgumentsModule::SOURCE_BINDING`]).
pub const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the scratch buffer ([`MeshArgumentsModule::SCRATCH_BINDING`]).
pub const SCRATCH: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the status buffer ([`MeshArgumentsModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};
/// The push field of the source residual, as a descriptor slot's residual bind offset.
const RESIDUAL: PushField = PushField::SlotBindOffset(SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Buffer,
    index: 0,
});

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// Whether `v`'s product is at most `total`, without overflow and without a division: a zero
/// count's product is 0, and otherwise the product is within where neither multiplication's high
/// word is set and its low word is at most `total` (`checked_mul`, as `bind::replay`'s).
fn total_within(f: &mut FunctionBuilder<'_>, v: [Value; 3], total: u32) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let t = u32_value(f, total)?;
    let mut any_zero = f.binary(BinOp::IEqual, v[0], zero)?;
    for &c in &v[1..] {
        let z = f.binary(BinOp::IEqual, c, zero)?;
        any_zero = f.binary(BinOp::LogicalOr, any_zero, z)?;
    }
    let (xy, over_xy) = checked_mul(f, v[0], v[1], zero)?;
    let (xyz, over_xyz) = checked_mul(f, xy, v[2], zero)?;
    let over = f.binary(BinOp::LogicalOr, over_xy, over_xyz)?;
    let le = f.binary(BinOp::ULessThanEqual, xyz, t)?;
    let fits = f.unary(UnOp::LogicalNot, over)?;
    let fits = f.binary(BinOp::LogicalAnd, fits, le)?;
    f.binary(BinOp::LogicalOr, any_zero, fits)
}

/// Whether each of `v` is at most its axis of `limits`, and their product at most its total.
fn grid_within(
    f: &mut FunctionBuilder<'_>,
    v: [Value; 3],
    limits: MeshGridLimits,
) -> BResult<Value> {
    let mut within = total_within(f, v, limits.total)?;
    for (c, axis) in v.into_iter().zip(limits.axes) {
        let a = u32_value(f, axis)?;
        let le = f.binary(BinOp::ULessThanEqual, c, a)?;
        within = f.binary(BinOp::LogicalAnd, within, le)?;
    }
    Ok(within)
}

/// The `bool` constant `true`.
fn truth(f: &mut FunctionBuilder<'_>) -> BResult<Value> {
    let ty = f.module().ty(Type::Bool)?;
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(1),
    })?;
    f.module().const_value(c)
}

/// The arguments module for `rule` as a typed module, before serialization: the module
/// [`mesh_arguments_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`SOURCE`], [`SCRATCH`] and [`STATUS`]; its push field is the source residual at
/// offset 0.
pub fn typed_module(rule: MeshGridRule) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let source = words_buffer(&mut b, SOURCE, u32_)?;
    let scratch = words_buffer(&mut b, SCRATCH, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let residual = b.push_member(RESIDUAL, u32_, 0)?;
    let push = b.push_global()?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let two = u32_value(&mut f, 2)?;
    let push_value = f.module().global_value(push)?;
    let member = u32_value(&mut f, residual.index)?;
    let residual_ptr = f.access_chain(push_value, &[member])?;
    let residual_bytes = f.load(residual_ptr, MemAccess::default())?;
    let first = f.binary(BinOp::ShiftRightLogical, residual_bytes, two)?;
    let source_value = f.module().global_value(source)?;
    let len = f.array_length(source_value, 0)?;
    let mut v = [zero; 3];
    for (axis, out) in v.iter_mut().enumerate() {
        let i = u32_value(&mut f, axis as u32)?;
        let index = f.binary(BinOp::IAdd, first, i)?;
        // Contained: an index past the bound range reads word 0 and yields 0.
        let inb = f.binary(BinOp::ULessThan, index, len)?;
        let safe = f.select(inb, index, zero)?;
        let p = f.access_chain(source_value, &[zero, safe])?;
        let loaded = f.load(p, MemAccess::default())?;
        *out = f.select(inb, loaded, zero)?;
    }
    let metal = match rule.metal {
        Some(limits) => grid_within(&mut f, v, limits)?,
        None => truth(&mut f)?,
    };
    let device = grid_within(&mut f, v, rule.device)?;
    let drawn = f.binary(BinOp::LogicalAnd, metal, device)?;
    let scratch_value = f.module().global_value(scratch)?;
    for (axis, c) in v.into_iter().enumerate() {
        let i = u32_value(&mut f, axis as u32)?;
        let count = f.select(drawn, c, zero)?;
        let out = f.access_chain(scratch_value, &[zero, i])?;
        f.store(out, count, MemAccess::default())?;
    }
    // Metal draws the grid, and the device's first stage cannot launch it.
    let past = f.binary(BinOp::LogicalNotEqual, metal, drawn)?;
    let status_value = f.module().global_value(status)?;
    let word = u32_value(&mut f, StatusWord::Status.index())?;
    let bit = u32_value(&mut f, StatusBits::MESH_GRID_PAST_LIMITS.0)?;
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let p = fb.access_chain(status_value, &[zero, word])?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), past, then, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(MeshArgumentsModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(MeshArgumentsModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SCRATCH => Some(MeshArgumentsModule::SCRATCH_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(MeshArgumentsModule::STATUS_BINDING),
        _ => None,
    }
}

/// The arguments module of `device` for draws whose first stage is `form`: built with the typed
/// builder, checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`MeshArgumentsModule`].
pub fn mesh_arguments_module(
    device: &Device,
    form: MeshArgumentsForm,
) -> Result<MeshArgumentsModule, Error> {
    let profile = device.profile();
    let rule = MeshGridRule::of(form, &profile.fields().limits.mesh);
    let m = typed_module(rule)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the mesh arguments pass needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(MeshArgumentsModule {
        spirv: words.into(),
        requirements: r.requirements,
        rule,
    })
}

#[cfg(test)]
mod tests;
