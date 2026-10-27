//! Program-scope bindings as entry parameters.
//!
//! MSL 3.2 lets a program-scope variable bind an argument-table slot (`texture2d<half> t
//! [[texture(n)]];`, `constant T* constant p [[buffer(n)]];`), which any function may read. `xcrun
//! metal` 32023.921 lists each in `!air.global_bindings` ([`GlobalBinding`]) with the metadata of
//! the entry parameter it stands for, and makes the variable a constant-space, externally
//! initialized `undef` global holding the slot's handle or pointer: a texture's is `{ ptr
//! addrspace(1) }`, which a read loads as `load ptr addrspace(1), ptr addrspace(2) @t`. The slot's
//! contents are what the entry's parameter of that role at that slot would hold, so after inlining
//! `expand` adds to the entry one parameter per bound global its body loads, of the loaded type,
//! with the binding's role, and makes each load of the global (through its address, a cast of it,
//! or a zero-index `getelementptr` of it) that parameter. Lowering then binds it as it binds every
//! parameter of its role. Any other use of a bound global (a load at another type than the first,
//! a store, its address taken as a value) is refused as `Unsupported{ProgramScopeBinding}`.

use crate::abi::{GlobalBinding, ParamBinding};
use crate::budget::Meter;
use crate::error::{Error, Feature, Location, Unsupported};
use crate::ir::{ArgIdx, Body, Const, ConstExpr, ConstId, ConstTable, FuncId, GlobalId};
use crate::ir::{Idx, InstId, InstKind, ModuleParts, TypeId, Value};

use super::rewrite::{append_params, apply, filled, invariant, scoped, BodyEdit};

/// The refusal of a use of a bound global this pass does not make a parameter.
fn refused() -> Error {
    let at = Location::metadata(None, None, Some("air.global_bindings"));
    Unsupported::not_implemented(Feature::ProgramScopeBinding)
        .at(at)
        .into()
}

/// The global whose address `v` is: the address itself, a cast of it, or a `getelementptr` of it
/// whose every index is zero (the address of its first field).
fn address_of(consts: &ConstTable, v: Value) -> Option<GlobalId> {
    let Value::Const(mut c) = v else {
        return None;
    };
    // A chain of casts and zero-index GEPs is short; the bound keeps a malformed one finite.
    for _ in 0..16 {
        match &consts.get(c)?.value {
            Const::GlobalAddr(g) => return Some(*g),
            Const::Expr(ConstExpr::Cast { ops: [inner], .. }) => c = *inner,
            Const::Expr(ConstExpr::Gep { ops, .. }) => {
                let (&base, indices) = ops.split_first()?;
                let zero = indices.iter().all(|&i| {
                    matches!(
                        consts.get(i).map(|k| &k.value),
                        Some(Const::Int(0) | Const::Null)
                    )
                });
                if !zero {
                    return None;
                }
                c = base;
            }
            _ => return None,
        }
    }
    None
}

/// Which constants of `parts` mention a bound global, at any depth: children precede parents, so
/// one walk in id order answers every constant. The table's bytes are added to `held`.
fn mentions(
    parts: &ModuleParts,
    bound: &dyn Fn(GlobalId) -> Option<usize>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Vec<bool>, Error> {
    let mut mentions = filled(m, held, parts.consts.len(), false)?;
    for (id, c) in parts.consts.iter() {
        m.tick(1)?;
        let direct = matches!(c.value, Const::GlobalAddr(g) if bound(g).is_some());
        let nested = c
            .value
            .children()
            .iter()
            .any(|k| mentions.get(k.index()).copied().unwrap_or(false));
        if let Some(slot) = mentions.get_mut(id.index()) {
            *slot = direct || nested;
        }
    }
    Ok(mentions)
}

/// Make each load of a bound global in the entry `func`'s inlined `body` a read of a parameter
/// added to the entry (see the module), returning the new body and the added parameters' bindings
/// (none, and `body` unchanged, when the module binds no global or the body reads none).
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    bindings: &[GlobalBinding],
    m: &mut Meter,
) -> Result<(Body, Vec<ParamBinding>), Error> {
    if bindings.is_empty() {
        return Ok((body, Vec::new()));
    }
    let bound = |g: GlobalId| bindings.iter().position(|b| b.global == g);
    // The bindings read, in the order the body first loads them, with the loaded type, and each
    // load of one. The table of the constants that mention a bound global is scratch.
    let (read, loads) = scoped(m, |m, held| {
        let mentions = mentions(parts, &bound, m, held)?;
        let mentioned = |c: ConstId| mentions.get(c.index()).copied().unwrap_or(false);
        let mut read: Vec<(usize, TypeId)> = Vec::new();
        let mut loads: Vec<(InstId, usize)> = Vec::new();
        let mut result = Ok(());
        for (i, inst) in body.insts.iter() {
            m.tick(1)?;
            if let InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } = &inst.kind
            {
                if let Some(at) = address_of(&parts.consts, *ptr).and_then(bound) {
                    match read.iter().find(|(b, _)| *b == at) {
                        Some(&(_, ty)) if ty != inst.ty => result = Err(refused()),
                        Some(_) => {}
                        None => {
                            m.reserve(&mut read, 1)?;
                            read.push((at, inst.ty));
                        }
                    }
                    m.reserve(&mut loads, 1)?;
                    loads.push((i, at));
                    continue;
                }
            }
            inst.kind.for_each_value(|v| {
                if let Value::Const(c) = v {
                    if mentioned(c) {
                        result = Err(refused());
                    }
                }
            });
        }
        result.map(|()| (read, loads))
    })?;
    if read.is_empty() {
        return Ok((body, Vec::new()));
    }
    let tys: Vec<TypeId> = read.iter().map(|&(_, ty)| ty).collect();
    let first = append_params(parts, func, &tys, m)?;
    let param = |at: usize| -> Option<u32> {
        let n = read.iter().position(|&(b, _)| b == at)?;
        first.checked_add(u32::try_from(n).ok()?)
    };
    let out = scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(&body, m, held)?;
        for &(i, at) in &loads {
            m.tick(1)?;
            let p = param(at).ok_or_else(|| invariant("an unread binding"))?;
            edit.replace(i, Value::Arg(ArgIdx::new(p)));
        }
        apply(&body, &edit, &|c| c, m)
    })?;
    body.release(m);
    let mut added = Vec::new();
    m.reserve(&mut added, read.len())?;
    for (n, &(at, _)) in read.iter().enumerate() {
        let b = bindings
            .get(at)
            .ok_or_else(|| invariant("binding out of range"))?;
        let p = first
            .checked_add(u32::try_from(n).map_err(|_| invariant("parameters"))?)
            .ok_or_else(|| invariant("parameters"))?;
        added.push(ParamBinding {
            param: p,
            role: b.role.clone(),
            unused: false,
            predicate: None,
            ty: b.ty.clone(),
            name: b.name.clone(),
        });
    }
    Ok((out, added))
}
