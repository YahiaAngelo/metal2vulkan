//! Texture block reads as plain texel reads.
//!
//! iOS 27's Metal compiler (metalfe-32023.921, AIR 2.7) emits
//! `void air.block_read_texture_2d.<T>(tex, smp, <2 x i32> coord, <2 x i32> offset, i32 lod,
//! i32 access, ptr out, ptr status, <2 x i32> size)`. Its first six operands are exactly a
//! current-form 2D texel read's (`air.read_texture_2d.<T>`: `{T, i8} (tex, smp, coord, offset,
//! lod, access)`, `intrinsics::texture`); the call adds the destination of the texels, an optional
//! destination of the residency status, and the block's extent. The reading this expansion takes,
//! from the IR of the shipped kernels that use it (a camera kernel reads a `2 x 1` block of a
//! `texture2d<half, access::sample>` into a `half4[2]` and pairs element `k` with the texel at
//! `coord + (k, 0)`; a second call at offset `(0, 1)` fills the next row):
//!
//! - element `y · size.x + x` of `out`, an array of `T` (`T`'s allocation size apart), is the texel
//!   `read(coord + (x, y), offset, lod)` with the call's access, for `x < size.x`, `y < size.y`
//!   (the first extent lane is the width, as the kernel's pairing shows);
//! - `status`, when not null, would receive the residency byte; no observed call passes one, so a
//!   call with a status pointer other than the null constant is not expanded;
//! - a call without a sampler (`poison` or `undef`, as the shipped kernels pass) reads as Metal's
//!   sampler-less `read` is built: each read takes the front end's read sampler
//!   (`air.get_read_sampler`) and a zero offset, the call's offset added to the coordinate (a
//!   sampler-less read takes no offset; Metal adds a read's offset to its coordinate, MSL 4.1
//!   §6.13). A call with a sampler passes it and the offset to each read, whose lowering takes or
//!   refuses them as it does a `read(sampler, coord, offset)`.
//!
//! [`expand`] makes each call whose `size` is a constant `(w, h)` with `1 ≤ w · h ≤ 64`, whose
//! access is a literal (the read binds it as an immediate) and whose status pointer is the null
//! constant, a call of a function built here that performs the `w · h` reads and stores in
//! row-major order, and inlines it. Anything else stays the unknown intrinsic it was, which the
//! post-check refuses (`UnknownIntrinsic`), as does a declaration whose name or type does not
//! match the read it mirrors (the read is classified as decode classifies it).

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{self, DeclClass, NumTy};
use crate::ir::{
    AddrSpace, ArgIdx, BinOp, Call, CallConv, Callee, Const, Constant, FastMath, FnAttrs, FuncId,
    Function, InstId, InstKind, Linkage, ModuleParts, Param, ParamAttrs, TailKind, Type, TypeId,
    UnnamedAddr, Value, Visibility,
};

use super::inline;
use super::out::Out;
use super::rewrite::invariant;

/// The declaration name's stem; the rest (`<T>`, with any `.i16` / sign tokens) is the read's.
const STEM: &str = "air.block_read_texture_2d.";
/// The texel read the block read mirrors.
const READ_STEM: &str = "air.read_texture_2d.";
/// The operands: `(tex, smp, coord, offset, lod, access, out, status, size)`.
const ARGS: usize = 9;
/// The largest block expanded, in texels: each is one read.
const MAX_TEXELS: u64 = 64;

/// One expansion: the declaration, the block's extent, and the access immediate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    decl: FuncId,
    w: u32,
    h: u32,
    access: u64,
    /// The call passes no sampler (`poison` or `undef`): each read takes the front end's read
    /// sampler, and the offset is added to the coordinate.
    sampler_less: bool,
}

/// A declaration of a block read: its read's name, the read's texel type token, and the
/// declaration's parameter types.
struct Decl {
    read: String,
    texel: NumTy,
    params: Box<[TypeId]>,
}

/// The block read `decl` declares, when its name and type are the module doc's form.
fn declaration(parts: &ModuleParts, decl: FuncId) -> Option<Decl> {
    let f = parts.funcs.get(decl)?;
    if f.body.is_some() {
        return None;
    }
    let name = std::str::from_utf8(parts.symbols.get(f.name)?).ok()?;
    let rest = name.strip_prefix(STEM)?;
    let texel = NumTy::parse(rest.rsplit('.').next()?)?;
    let Some(Type::Function {
        ret,
        params,
        vararg: false,
    }) = parts.types.get(f.ty)
    else {
        return None;
    };
    if params.len() != ARGS || !matches!(parts.types.get(*ret), Some(Type::Void)) {
        return None;
    }
    let ty = |i: usize| params.get(i).and_then(|&t| parts.types.get(t));
    let thread_ptr = |i: usize| {
        matches!(
            ty(i),
            Some(Type::Ptr {
                space: AddrSpace::Thread
            })
        )
    };
    // The coordinate and offset share one integer vector type of two lanes; the extent is a
    // `<2 x i32>`.
    let int2 = |t: Option<&Type>| match t {
        Some(Type::Vector { elem, len: 2 }) => {
            matches!(parts.types.get(*elem), Some(Type::Int(16 | 32)))
        }
        _ => false,
    };
    let coord_same = params.get(2) == params.get(3);
    let size_ok = match ty(8) {
        Some(Type::Vector { elem, len: 2 }) => {
            matches!(parts.types.get(*elem), Some(Type::Int(32)))
        }
        _ => false,
    };
    if !(int2(ty(2)) && coord_same && thread_ptr(6) && thread_ptr(7) && size_ok) {
        return None;
    }
    Some(Decl {
        read: format!("{READ_STEM}{rest}"),
        texel,
        params: params.clone(),
    })
}

/// The literal integer `v`, when it is one.
fn literal(parts: &ModuleParts, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match parts.consts.get(c)?.value {
        Const::Int(x) => Some(x),
        _ => None,
    }
}

/// The literal `<2 x i32>` lanes of `v`.
fn pair(parts: &ModuleParts, v: Value) -> Option<(u32, u32)> {
    let Value::Const(c) = v else { return None };
    match &parts.consts.get(c)?.value {
        Const::Aggregate(lanes) => match **lanes {
            [a, b] => {
                let lane = |c| match parts.consts.get(c)?.value {
                    Const::Int(x) => u32::try_from(x).ok(),
                    _ => None,
                };
                Some((lane(a)?, lane(b)?))
            }
            _ => None,
        },
        _ => None,
    }
}

/// How call `call` of an unknown intrinsic is expanded, or `None` when it is not (module doc).
fn site(parts: &ModuleParts, call: &Call) -> Option<Key> {
    let Callee::UnknownIntrinsic(decl) = call.callee else {
        return None;
    };
    declaration(parts, decl)?;
    if call.args.len() != ARGS {
        return None;
    }
    let access = literal(parts, *call.args.get(5)?)?;
    let status = match *call.args.get(7)? {
        Value::Const(c) => parts.consts.get(c)?.value == Const::Null,
        _ => false,
    };
    let sampler_less = match *call.args.get(1)? {
        Value::Const(c) => matches!(parts.consts.get(c)?.value, Const::Poison | Const::Undef),
        _ => false,
    };
    let (w, h) = pair(parts, *call.args.get(8)?)?;
    let texels = u64::from(w) * u64::from(h);
    (status && (1..=MAX_TEXELS).contains(&texels)).then_some(Key {
        decl,
        w,
        h,
        access,
        sampler_less,
    })
}

/// Whether `parts` declares a block read; a module that declares none has no site, and [`expand`]
/// need not read its instructions.
fn declares_sites(parts: &ModuleParts, m: &mut Meter) -> Result<bool, Error> {
    for f in parts.funcs.values() {
        m.tick(1)?;
        if f.body.is_none()
            && parts
                .symbols
                .get(f.name)
                .is_some_and(|n| n.starts_with(STEM.as_bytes()))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Make each block read of the inlined entry `func`'s `body` that the module doc lists a call of a
/// function built for it, and inline the result.
pub(super) fn expand(
    body: crate::ir::Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<crate::ir::Body, Error> {
    if !declares_sites(parts, m)? {
        return Ok(body);
    }
    let mut plans: Vec<(InstId, Key)> = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        if let Some(key) = site(parts, call) {
            m.reserve(&mut plans, 1)?;
            plans.push((id, key));
        }
    }
    if plans.is_empty() {
        m.release(plans.capacity() * std::mem::size_of::<(InstId, Key)>());
        return Ok(body);
    }
    // One function per key; a key whose read the decoder does not know leaves its calls alone.
    let mut made: Vec<(Key, Option<FuncId>)> = Vec::new();
    let mut calls: Vec<(InstId, FuncId)> = Vec::new();
    m.reserve(&mut calls, plans.len())?;
    for &(id, key) in &plans {
        let f = match made.iter().find(|(k, _)| *k == key) {
            Some(&(_, f)) => f,
            None => {
                let f = function(parts, key, m)?;
                m.reserve(&mut made, 1)?;
                made.push((key, f));
                f
            }
        };
        if let Some(f) = f {
            calls.push((id, f));
        }
    }
    m.release(made.capacity() * std::mem::size_of::<(Key, Option<FuncId>)>());
    drop(made);
    m.release(plans.capacity() * std::mem::size_of::<(InstId, Key)>());
    drop(plans);
    if calls.is_empty() {
        m.release(calls.capacity() * std::mem::size_of::<(InstId, FuncId)>());
        return Ok(body);
    }
    let mut body = body;
    for &(id, f) in &calls {
        m.tick(1)?;
        let inst = body
            .insts
            .get_mut(id)
            .ok_or_else(|| invariant("a planned block read out of range"))?;
        let InstKind::Call(call) = &mut inst.kind else {
            return Err(invariant("a planned block read is not a call"));
        };
        let lowered = Call {
            callee: Callee::Direct(f),
            args: call.args.clone(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        };
        m.charge(lowered.heap_bytes())?;
        m.release(call.heap_bytes());
        **call = lowered;
    }
    m.release(calls.capacity() * std::mem::size_of::<(InstId, FuncId)>());
    drop(calls);
    if let Some(replaced) = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?
        .body
        .replace(body)
    {
        replaced.release(m);
    }
    inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        func,
        m,
        |_, _| Ok(()),
    )?
    .ok_or_else(|| invariant("an expanded block read was not inlined"))
}

/// The IR type of texel `t`.
fn texel_type(out: &mut Out<'_>, t: NumTy, m: &mut Meter) -> Result<TypeId, Error> {
    use crate::intrinsics::Elem;
    let scalar = match t.elem {
        Elem::F16 => Type::Half,
        Elem::F32 => Type::Float,
        Elem::BF16 => Type::BFloat,
        e => Type::Int(u8::try_from(e.bits()).map_err(|_| invariant("an element's width"))?),
    };
    let scalar = out.ty(scalar, m)?;
    if t.lanes == 1 {
        return Ok(scalar);
    }
    out.ty(
        Type::Vector {
            elem: scalar,
            len: t.lanes,
        },
        m,
    )
}

/// The function of `key`: its parameters are the call's operands. `None` when the read it
/// mirrors is not one the decoder knows.
fn function(parts: &mut ModuleParts, key: Key, m: &mut Meter) -> Result<Option<FuncId>, Error> {
    let decl = declaration(parts, key.decl)
        .ok_or_else(|| invariant("a planned block read's declaration"))?;
    let fn_ty = parts
        .funcs
        .get(key.decl)
        .map(|f| f.ty)
        .ok_or_else(|| invariant("a declaration out of range"))?;
    let read_params = decl
        .params
        .get(..6)
        .ok_or_else(|| invariant("a block read's parameters"))?;
    // The read's declaration must classify as the decoder's texel read of these types.
    {
        let i8t = parts.types.intern(Type::Int(8), m)?;
        let tex = {
            let mut o = Out::new(parts);
            texel_type(&mut o, decl.texel, m)?
        };
        let ret = parts.types.intern(
            Type::Struct {
                fields: Box::new([tex, i8t]),
                packed: false,
                name: None,
            },
            m,
        )?;
        let ty = parts.types.intern(
            Type::Function {
                ret,
                params: read_params.into(),
                vararg: false,
            },
            m,
        )?;
        let sym = parts.symbols.intern(decl.read.as_bytes(), m)?;
        let class = intrinsics::classify(decl.read.as_bytes(), sym, ty, &parts.types);
        if !matches!(class, Some(c) if !matches!(c, DeclClass::Unknown | DeclClass::Mismatch)) {
            return Ok(None);
        }
    }
    let arg = |i: u32| Value::Arg(ArgIdx::new(i));
    let mut out = Out::new(parts);
    out.open(None)?;
    let tex = texel_type(&mut out, decl.texel, m)?;
    let i8t = out.int_ty(8, m)?;
    let ret = out.ty(
        Type::Struct {
            fields: Box::new([tex, i8t]),
            packed: false,
            name: None,
        },
        m,
    )?;
    let coord_ty = *read_params
        .get(2)
        .ok_or_else(|| invariant("a block read's coordinate"))?;
    let lane_ty = match out.parts.types.get(coord_ty) {
        Some(Type::Vector { elem, .. }) => *elem,
        _ => return Err(invariant("a block read's coordinate is not a vector")),
    };
    let access_ty = *read_params
        .get(5)
        .ok_or_else(|| invariant("a block read's access"))?;
    let access = out.constant(access_ty, Const::Int(key.access), m)?;
    let i64t = out.int_ty(64, m)?;
    // A sampler-less block read reads as Metal's sampler-less `read` is built: through the read
    // sampler, which addresses nothing and takes no offset, at the coordinate plus the offset.
    let (sampler, base, offset) = if key.sampler_less {
        let smp_ty = *read_params
            .get(1)
            .ok_or_else(|| invariant("a block read's sampler"))?;
        let smp = out.intrinsic("air.get_read_sampler", smp_ty, &[], &[], m)?;
        let at = out.bin(coord_ty, BinOp::Add, arg(2), arg(3), m)?;
        let zero = out.constant(coord_ty, Const::Null, m)?;
        (smp, at, zero)
    } else {
        (arg(1), arg(2), arg(3))
    };
    for y in 0..key.h {
        for x in 0..key.w {
            m.tick(1)?;
            let coord = if (x, y) == (0, 0) {
                base
            } else {
                let lane = |out: &mut Out<'_>, v: u32, m: &mut Meter| {
                    out.parts.consts.intern(
                        Constant {
                            ty: lane_ty,
                            value: Const::Int(u64::from(v)),
                        },
                        m,
                    )
                };
                let (a, b) = (lane(&mut out, x, m)?, lane(&mut out, y, m)?);
                let step = out.constant(coord_ty, Const::Aggregate(Box::new([a, b])), m)?;
                out.bin(coord_ty, BinOp::Add, base, step, m)?
            };
            let args = [arg(0), sampler, coord, offset, arg(4), access];
            let r = out.intrinsic(&decl.read, ret, read_params, &args, m)?;
            let texel = out.push(
                tex,
                InstKind::ExtractValue {
                    agg: r,
                    indices: Box::new([0]),
                },
                m,
            )?;
            let k = u64::from(y) * u64::from(key.w) + u64::from(x);
            let ptr = if k == 0 {
                arg(6)
            } else {
                let ptr_ty = out.ptr_ty(AddrSpace::Thread, m)?;
                let k = out.constant(i64t, Const::Int(k), m)?;
                out.push(
                    ptr_ty,
                    InstKind::Gep {
                        src_ty: tex,
                        base: arg(6),
                        indices: Box::new([k]),
                        inbounds: true,
                    },
                    m,
                )?
            };
            let align = u64::from(decl.texel.elem.bits() / 8);
            out.store(texel, ptr, align, m)?;
        }
    }
    out.ret(None, m)?;
    let Out { body, parts, .. } = out;
    let name = format!("__m2v.block_read.{}", parts.funcs.len());
    let name = parts.symbols.intern(name.as_bytes(), m)?;
    let mut ps = Vec::new();
    m.reserve(&mut ps, ARGS)?;
    ps.resize(ARGS, Param::default());
    let f = Function {
        name,
        ty: fn_ty,
        linkage: Linkage::Internal,
        visibility: Visibility::Default,
        unnamed_addr: UnnamedAddr::Significant,
        cconv: CallConv::C,
        attrs: FnAttrs::default(),
        ret_attrs: ParamAttrs::default(),
        params: ps.into_boxed_slice(),
        body: Some(body),
        section: None,
        align: None,
    };
    Ok(Some(parts.funcs.push(f, m)?))
}
