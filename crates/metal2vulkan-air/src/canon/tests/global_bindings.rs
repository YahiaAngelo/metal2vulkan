//! Program-scope bindings become entry parameters: each load of a global `!air.global_bindings`
//! binds, in the entry or a function it inlines, is a read of a parameter of the binding's role
//! appended to the entry, one per global read.

use super::*;
use crate::abi::{GlobalBinding, TextureArg};
use crate::ir::{InstKind, Type, TypeId};

/// A program-scope binding's global: a constant-space, externally initialized `undef` object of
/// `value_ty`, as `xcrun metal` 32023.921 writes one.
fn bound_global(m: &mut ModuleBuilder, name: &str, value_ty: TypeId) -> GlobalId {
    let undef = m.constant(value_ty, Const::Undef);
    let name = m.sym(name);
    m.global(Global {
        name,
        value_ty,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(undef),
        externally_initialized: true,
        linkage: Linkage::Internal,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        align: None,
        section: None,
    })
}

fn texture_binding(global: GlobalId, slot: u32) -> GlobalBinding {
    GlobalBinding {
        global,
        role: ParamRole::Texture(TextureArg {
            location: Slot {
                index: AbiInt::Const(slot),
                count: AbiInt::Const(1),
            },
            access: Access::Sample,
            raster_order_group: None,
        }),
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new("lut")),
    }
}

fn buffer_binding(global: GlobalId, slot: u32) -> GlobalBinding {
    let ParamRole::Buffer(mut arg) = buffer(0, slot).role else {
        unreachable!()
    };
    arg.access = Access::Read;
    arg.space = AddrSpace::Constant;
    GlobalBinding {
        global,
        role: ParamRole::Buffer(arg),
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new("table")),
    }
}

/// `kernel k(device i32* out [[buffer(0)]])` with a program-scope `texture2d<half> lut
/// [[texture(5)]]` and `constant i32* constant table [[buffer(3)]]`: `body` builds the value
/// stored to `out` from the two globals.
fn entry(
    body: impl FnOnce(&mut FunctionBuilder<'_>, GlobalId, GlobalId) -> Value,
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let dev = m.ptr(AddrSpace::Device);
    let cst = m.ptr(AddrSpace::Constant);
    let tex_ty = m.struct_ty(&[dev]);
    let lut = bound_global(&mut m, "lut", tex_ty);
    let table = bound_global(&mut m, "table", cst);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = body(&mut f, lut, table);
    store(&mut f, v, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let mut abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    abi.global_bindings = vec![texture_binding(lut, 5), buffer_binding(table, 3)];
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// The width of the texture `tex` (`air.get_width_texture_2d`).
fn width(f: &mut FunctionBuilder<'_>, tex: Value) -> Value {
    let i32t = f.module().int(32);
    let dev = f.module().ptr(AddrSpace::Device);
    let zero = f.module().const_int(i32t, 0);
    let name = "air.get_width_texture_2d";
    let ty = f.module().fn_ty(i32t, &[dev, i32t]);
    let decl = f.module().declare(name, ty);
    let op = AirIntrinsic::parse(name).unwrap();
    call(f, i32t, Callee::Air { decl, op }, &[tex, zero])
}

/// The entry's parameter types, printed.
fn param_types(e: &Entry) -> Vec<String> {
    let f = e.module.func(e.func).unwrap();
    let Some(Type::Function { params, .. }) = e.module.types().get(f.ty) else {
        panic!("the entry's type is not a function");
    };
    params
        .iter()
        .map(|t| format!("{:?}", e.module.types().get(*t).unwrap()))
        .collect()
}

/// A texture handle loaded from its global, twice, and a buffer pointer loaded from its global in
/// a helper the entry calls are one parameter each, appended in the order the inlined body first
/// reads them, with their bindings' roles; no load of either global is left.
#[test]
fn bound_globals_become_parameters() {
    let e = entry(|f, lut, table| {
        let i32t = f.module().int(32);
        let dev = f.module().ptr(AddrSpace::Device);
        let cst = f.module().ptr(AddrSpace::Constant);
        // The helper reads the bound buffer: inlining brings its load into the entry.
        let hty = f.module().fn_ty(i32t, &[]);
        let table_addr = f.module().global_addr(table);
        let mut h = f.module().define("helper", hty);
        let p = load(&mut h, cst, table_addr);
        let x = load(&mut h, i32t, p);
        h.ret(Some(x));
        let helper = h.finish();
        let lut_addr = f.module().global_addr(lut);
        let t0 = load(f, dev, lut_addr);
        let t1 = load(f, dev, lut_addr);
        let w0 = width(f, t0);
        let w1 = width(f, t1);
        let x = call(f, i32t, Callee::Direct(helper), &[]);
        let w = f.bin(BinOp::Add, w0, w1);
        f.bin(BinOp::Add, w, x)
    })
    .unwrap();
    let roles: Vec<_> =
        e.ep.params
            .iter()
            .map(|p| (p.param, p.role.clone()))
            .collect();
    assert_eq!(roles.len(), 3, "{roles:?}");
    assert_eq!(roles[1], (1, texture_binding(GlobalId::new(0), 5).role));
    assert_eq!(roles[2], (2, buffer_binding(GlobalId::new(0), 3).role));
    let tys = param_types(&e);
    assert_eq!(
        tys,
        [
            "Ptr { space: Device }",
            "Ptr { space: Device }",
            "Ptr { space: Constant }"
        ]
    );
    let text = body_text(&e);
    assert!(!text.contains("@lut") && !text.contains("@table"), "{text}");
    assert_eq!(text.matches("](%a1, i32 0)").count(), 2, "{text}");
    assert!(text.contains("load i32, %a2"), "{text}");
    assert!(e.module.parts().globals.is_empty(), "{text}");
}

/// A module whose entry reads no bound global keeps the entry's parameters.
#[test]
fn an_unread_binding_adds_no_parameter() {
    let e = entry(|f, _, _| {
        let i32t = f.module().int(32);
        f.module().const_int(i32t, 7)
    })
    .unwrap();
    assert_eq!(e.ep.params.len(), 1);
    assert_eq!(param_types(&e).len(), 1);
}

/// A use of a bound global other than a load of it (here a store of its address) is refused as
/// not implemented, as is a load of it at a second type.
#[test]
fn other_uses_of_a_bound_global_are_refused() {
    let unsupported = |r: Result<Entry, Error>| match r {
        Err(Error::Unsupported(u)) => assert_eq!(u.feature, Feature::ProgramScopeBinding),
        other => panic!("not refused: {other:?}"),
    };
    unsupported(entry(|f, lut, _| {
        let i32t = f.module().int(32);
        let lut_addr = f.module().global_addr(lut);
        let thread = f.module().ptr(AddrSpace::Thread);
        let cst = f.module().ptr(AddrSpace::Constant);
        let one = f.module().const_int(i32t, 1);
        let slot = f.push(
            thread,
            InstKind::Alloca {
                ty: cst,
                count: one,
                align: None,
            },
        );
        store(f, lut_addr, slot);
        let p = load(f, cst, slot);
        let dev = f.module().ptr(AddrSpace::Device);
        let t = load(f, dev, p);
        width(f, t)
    }));
    unsupported(entry(|f, lut, _| {
        let i64t = f.module().int(64);
        let dev = f.module().ptr(AddrSpace::Device);
        let lut_addr = f.module().global_addr(lut);
        let t = load(f, dev, lut_addr);
        let w = width(f, t);
        let raw = load(f, i64t, lut_addr);
        let w64 = f.push(
            i64t,
            InstKind::Cast {
                op: CastOp::ZExt,
                v: w,
            },
        );
        let sum = f.bin(BinOp::Add, raw, w64);
        let i32t = f.module().int(32);
        f.push(
            i32t,
            InstKind::Cast {
                op: CastOp::Trunc,
                v: sum,
            },
        )
    }));
}
