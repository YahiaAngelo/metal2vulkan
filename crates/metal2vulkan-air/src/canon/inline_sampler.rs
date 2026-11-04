//! Constexpr samplers written inline as their state word.
//!
//! Metal's frontend writes a `constexpr sampler` as a constant global listed in
//! `!air.sampler_states` (`abi::sampler`), and a sampling takes the global's address. Older
//! frontends (macOS 10.15's SDK, AIR 2.2) also write it inline: the sampler operand of the
//! sampling call is the one-word state itself, `inttoptr (i64 W to ptr addrspace(2))`, with no
//! global and no `!air.sampler_states` entry. The word is the one-word form `abi::sampler` decodes
//! (bit 63 set, which carries no property).
//!
//! This pass gives each distinct such word a global of its own: an internal constant `i64`
//! holding `W`, decoded as a listed one would be, and the sampler operand of each call that names
//! the word becomes the global's address. The sampler is then the constexpr sampler it encodes,
//! everywhere a listed one is (numbering, binding, its state's checks). Only a sampler operand,
//! a position the intrinsic's typed signature marks [`HandleKind::Sampler`], is read so: the same
//! constant anywhere else stays as written, and lowering refuses it as before. A word that does
//! not decode is `AbiError::BadSamplerState`, never a guessed state.
//!
//! The new samplers follow the module's listed ones, in the order their first use appears in the
//! entry's body. Which number Metal's pipeline gives such a sampler is unmeasured: no such
//! number is observable before a pipeline reflects it, and the order here is a convention.

use crate::abi::{SamplerState, StaticSampler};
use crate::budget::Meter;
use crate::error::{AbiError, Error, InputError, InputErrorKind, Location};
use crate::intrinsics::{HandleKind, Operand};
use crate::ir::{AddrSpace, Align, Body, Callee, CastOp, Const, ConstExpr, ConstId, Constant};
use crate::ir::{Global, GlobalId, InstId, InstKind, Linkage, ModuleParts, Type, TypeId, Value};

/// The state word of `c` when it is `inttoptr (i64 W to ptr addrspace(2))`.
fn inline_word(parts: &ModuleParts, c: ConstId) -> Option<u64> {
    let k = parts.consts.get(c)?;
    if parts.types.ptr_space(k.ty) != Some(AddrSpace::Constant)
        || !matches!(parts.types.get(k.ty), Some(Type::Ptr { .. }))
    {
        return None;
    }
    let Const::Expr(ConstExpr::Cast {
        op: CastOp::IntToPtr,
        ops: [w],
    }) = &k.value
    else {
        return None;
    };
    let w = parts.consts.get(*w)?;
    match (parts.types.get(w.ty), &w.value) {
        (Some(Type::Int(64)), Const::Int(v)) => Some(*v),
        _ => None,
    }
}

/// Give each inline sampler word that `body`'s sampling calls take a constant global of its own,
/// and make those operands the globals' addresses (module docs). Returns the body and the
/// samplers made, in first-use order; `body` itself when there is none.
pub(super) fn expand(
    mut body: Body,
    parts: &mut ModuleParts,
    m: &mut Meter,
) -> Result<(Body, Vec<StaticSampler>), Error> {
    // Each call's sampler operands that are inline words: (instruction, operand, constant, word).
    let mut sites: Vec<(InstId, usize, ConstId, u64)> = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        let Callee::Air { op, .. } = &call.callee else {
            continue;
        };
        let sig = op.signature();
        for (i, p) in sig.params().iter().enumerate() {
            if !matches!(p, Operand::Handle(HandleKind::Sampler)) {
                continue;
            }
            if let Some(&Value::Const(c)) = call.args.get(i) {
                if let Some(w) = inline_word(parts, c) {
                    m.reserve(&mut sites, 1)?;
                    sites.push((id, i, c, w));
                }
            }
        }
    }
    if sites.is_empty() {
        return Ok((body, Vec::new()));
    }
    let i64t = parts.types.intern(Type::Int(64), m)?;
    let mut made: Vec<(u64, ConstId, ConstId)> = Vec::new();
    let mut samplers = Vec::new();
    for &(id, i, c, w) in &sites {
        m.tick(1)?;
        let addr = match made.iter().find(|&&(_, k, _)| k == c) {
            Some(&(_, _, addr)) => addr,
            None => {
                let bad = || {
                    Error::InvalidInput(InputError::new(
                        InputErrorKind::Abi(AbiError::BadSamplerState),
                        Location::metadata(None, None, None),
                    ))
                };
                // The one-word form sets bit 63; it carries no sampler property.
                let state = SamplerState::from_words([w & !(1 << 63), 0]).ok_or_else(bad)?;
                let g = global(parts, i64t, w, samplers.len(), m)?;
                let ty = parts.consts.get(c).map(|k| k.ty).ok_or_else(bad)?;
                let addr = parts.consts.intern(
                    Constant {
                        ty,
                        value: Const::GlobalAddr(g),
                    },
                    m,
                )?;
                m.reserve(&mut made, 1)?;
                made.push((w, c, addr));
                m.reserve(&mut samplers, 1)?;
                samplers.push(StaticSampler { global: g, state });
                addr
            }
        };
        if let Some(InstKind::Call(call)) = body.insts.get_mut(id).map(|x| &mut x.kind) {
            if let Some(slot) = call.args.get_mut(i) {
                *slot = Value::Const(addr);
            }
        }
    }
    m.release(sites.capacity() * std::mem::size_of::<(InstId, usize, ConstId, u64)>());
    m.release(made.capacity() * std::mem::size_of::<(u64, ConstId, ConstId)>());
    Ok((body, samplers))
}

/// The internal constant `i64` global holding sampler word `w`, the `n`th this pass makes.
fn global(
    parts: &mut ModuleParts,
    i64t: TypeId,
    w: u64,
    n: usize,
    m: &mut Meter,
) -> Result<GlobalId, Error> {
    let init = parts.consts.intern(
        Constant {
            ty: i64t,
            value: Const::Int(w),
        },
        m,
    )?;
    // The name is a diagnostic only: nothing branches on it.
    let name = parts
        .symbols
        .intern(format!("__air_inline_sampler_state.{n}").as_bytes(), m)?;
    parts.globals.push(
        Global {
            name,
            value_ty: i64t,
            space: AddrSpace::Constant,
            constant: true,
            init: Some(init),
            externally_initialized: false,
            linkage: Linkage::Internal,
            visibility: Default::default(),
            unnamed_addr: Default::default(),
            align: Align::from_bytes(8),
            section: None,
        },
        m,
    )
}
