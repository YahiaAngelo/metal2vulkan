//! A texture read through a snapshot at a slot that has a second alias binding beside its image.
//!
//! The copy that fills a snapshot (`lower::fragment`'s `snapshot`) fetches through one view, and
//! the read the snapshot stands for must be the fetch through that same view: the slot's raw view
//! where the slot's reads take the raw codes (an 8-bit sRGB texel's, whose view the binder names
//! the snapshot's form by), else the slot's own image.
//!
//! A slot's unsigned integer alias (a 16-bit float texel's own 16 bits, `HalfTexelBitsPolicy::
//! Exact`) shares that one `ResourceKind::RawTexture` binding and is not such a view: a read of
//! such a slot fetches the slot's image and reads the alias beside it, under the bound format's
//! class. The copy takes that same branch in its own arms, so the snapshot holds the texel's own
//! 16 bits where the class names 16-bit float channels and the image's encode elsewhere.
//!
//! - Under `DevicePolicy::exact_answers` a `texture2d<float, access::read>` slot takes that alias,
//!   and every form the binder can name for it — the bound view's codes, or the fetched lanes'
//!   bits — translates and passes L3, as it does under the fast default. An 8-bit sRGB texture,
//!   whose codes the reads do take, keeps its raw view on both.
//! - The copy fetches the alias under `exact_answers` wherever the form holds a 16-bit code: two
//!   fetches in the module, the copy's two exclusive arms, one through an image of unsigned
//!   integer sampled type. One fetch, of the slot's float image, under the fast default and at a
//!   form of narrower codes.
//! - Every format whose class names 16-bit float channels names `SnapshotForm::Half` or
//!   `SnapshotForm::Bits`, the two the arm stores the texel's own 16 bits in, so the arm stands
//!   wherever the class can select it.
//! - The copy pairs the form with its fetch's lanes where the invariant lives: a form whose codes
//!   those lanes do not hold is named, not left to the builder to refuse (a float code form over
//!   integer lanes walks into `FClamp`, and an integer form over float lanes bitcasts the value).
//!   Every one of the ten forms, over float lanes and over unsigned integer lanes.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips a test visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{BindState, BoundTexture, TextureFacts};
use crate::budget::Budget;
use crate::device::{Device, DevicePolicy};
use crate::library::Library;
use crate::metal::{PixelFormat, TextureType, TextureUsage};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::layout::BindingSource;
use crate::pipeline::sym::{SlotClass, SlotRef, SnapshotForm, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;

/// The snapshot slot under test.
const SLOT: SlotRef = SlotRef {
    stage: Stage::Fragment,
    class: SlotClass::Texture,
    index: 0,
};

/// Every form a snapshot can keep its texels in, in declaration order.
const FORMS: [SnapshotForm; 10] = [
    SnapshotForm::Bits,
    SnapshotForm::Unorm8,
    SnapshotForm::Snorm8,
    SnapshotForm::Unorm16,
    SnapshotForm::Snorm16,
    SnapshotForm::Half,
    SnapshotForm::Uint8,
    SnapshotForm::Sint8,
    SnapshotForm::Uint16,
    SnapshotForm::Sint16,
];

/// Whether `form`'s codes hold every lane an image whose sampled type is `f32` (`float_lanes`)
/// or `u32` returns: a float lane's every form but an integer one's codes, an unsigned integer
/// lane's the bits and the unsigned integer codes. The bits hold any lanes.
fn holds(form: SnapshotForm, float_lanes: bool) -> bool {
    match float_lanes {
        true => !matches!(
            form,
            SnapshotForm::Uint8 | SnapshotForm::Uint16 | SnapshotForm::Sint8 | SnapshotForm::Sint16
        ),
        false => matches!(
            form,
            SnapshotForm::Bits | SnapshotForm::Uint8 | SnapshotForm::Uint16
        ),
    }
}

/// The profiles whose draws can take a snapshot (the pixel interlock, fragment stores, the
/// feedback loop layout and dynamic color write masks); `vk12-floor` has none, and is left out.
const PROFILES: [&str; 4] = [
    "intel-irisxe-anv",
    "amd-780m-radv",
    "nvidia-rtx5080-laptop",
    "lavapipe",
];

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A fragment function returning the `uint4` texel of a `texture2d<uint, access::read>` slot under
/// its own pixel, the read an integer-lane snapshot slot stands for.
const UINT_READ: &str = r#"define <4 x i32> @sx_read_u(<4 x float> %0, ptr addrspace(1) %1) {
  %3 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %4 = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %3)
  %5 = call ptr addrspace(2) @air.get_read_sampler()
  %6 = call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %1, ptr addrspace(2) %5, <2 x i32> %4, <2 x i32> zeroinitializer, i32 0, i32 1)
  %7 = extractvalue { <4 x i32>, i8 } %6, 0
  ret <4 x i32> %7
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.fragment = !{!0}
!0 = !{ptr @sx_read_u, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"t"}
"#;

/// Bitcode of the owned fixture `name`, or `None` (skipped visibly) without `llvm-as`.
fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// Bitcode of the module `text`, or `None` (skipped visibly) without `llvm-as`.
fn assemble(text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-snapshot-exact-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("f.ll");
    std::fs::write(&ll, format!("{HEADER}{text}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the fragment: {why}"),
    }
}

/// Function `name` of `bc` as a stage source.
fn function(bc: &[u8], name: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(
        Library::from_air(bc)
            .expect("library")
            .function(name)
            .expect("function"),
    ))
}

/// A draw state that asks for a snapshot of texture slot 0 with `pixel_format` bound there.
fn state(pixel_format: Option<PixelFormat>) -> BindState {
    let mut s = BindState {
        snapshot_textures: [0].into(),
        ..Default::default()
    };
    if let Some(pf) = pixel_format {
        let mut facts = TextureFacts::new(TextureType::Type2D, pf);
        facts.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        s.stages
            .entry(Stage::Fragment)
            .or_default()
            .textures
            .insert(
                0,
                BoundTexture::Texture {
                    facts,
                    resource_id: None,
                },
            );
    }
    s
}

/// A `texture2d<float, access::read>` slot the draw reads through a snapshot translates under
/// every form the binder names for it, on the fast default and under `exact_answers`, where the
/// slot takes an unsigned integer alias binding beside its image. An 8-bit sRGB texture, whose
/// raw codes the read does take, keeps its raw view on both. Each variant passes L3 and binds the
/// snapshot as a storage image.
#[test]
fn a_snapshot_slot_with_an_integer_alias_beside_its_image_translates_under_every_form() {
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let mut d = RenderPipelineDescriptor::new(function(&bc, "tob_rect"));
    d.fragment = Some(function(&bc, "tob_read"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    // Each form `SnapshotForm::of` names for a float slot, by a texture the binder names it for.
    let bound = [
        (PixelFormat::RGBA8Unorm, SnapshotForm::Unorm8),
        (PixelFormat::RGBA8Snorm, SnapshotForm::Snorm8),
        (PixelFormat::RGBA16Unorm, SnapshotForm::Unorm16),
        (PixelFormat::RGBA16Snorm, SnapshotForm::Snorm16),
        (PixelFormat::RGBA16Float, SnapshotForm::Half),
        (PixelFormat::R16Float, SnapshotForm::Half),
        (PixelFormat::RGBA32Float, SnapshotForm::Bits),
        (PixelFormat::RGBA8UnormSRGB, SnapshotForm::Unorm8),
    ];
    let mut aliased = 0;
    for stem in PROFILES {
        for (label, policy) in [
            ("the fast default", DevicePolicy::default()),
            ("exact answers", DevicePolicy::default().exact_answers()),
        ] {
            let desc = crate::pipeline::testing::description(stem).expect("description");
            let dev = Device::new(desc, policy).expect("device");
            let base = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}/{label}: {e}"));
            for (pf, form) in bound {
                let plan = base
                    .plan(&dev, &state(Some(pf)).shape())
                    .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: plan: {e}"));
                assert_eq!(
                    plan.variant.snapshot_textures,
                    [(SLOT, form)].into(),
                    "{stem}/{label}/{pf:?}"
                );
                let variant = dev
                    .compile_render_variant(&d, &plan.variant, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: {e}"));
                assert!(
                    variant
                        .layout
                        .sets
                        .iter()
                        .flat_map(|s| &s.bindings)
                        .any(|b| b.source == BindingSource::SnapshotTexels(0)),
                    "{stem}/{label}/{pf:?}: no snapshot binding"
                );
                // The alias binding beside the slot's image, the leg under test: under
                // `exact_answers` a `float` read takes it, and an sRGB slot's raw view is that
                // one binding on both policies.
                let raw = variant
                    .layout
                    .sets
                    .iter()
                    .flat_map(|s| &s.bindings)
                    .any(|b| b.source == BindingSource::RawTexture(SLOT));
                let srgb = pf == PixelFormat::RGBA8UnormSRGB;
                let exact = label == "exact answers";
                assert_eq!(raw, srgb || exact, "{stem}/{label}/{pf:?}: raw view");
                aliased += u32::from(raw && !srgb);
                for m in &variant.stages {
                    check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: L3: {e}"));
                }
            }
        }
    }
    assert!(aliased >= 7, "{aliased} variants took the alias binding");
}

/// The copy of a snapshot slot names a form whose codes its fetch's lanes do not hold, where that
/// invariant lives, instead of leaving the builder to refuse the encode: over float lanes every
/// integer form, and over unsigned integer lanes every float code form (the pairing that otherwise
/// reaches `FClamp`). The binder never names such a pairing (`SnapshotForm::of`), so the key is
/// forged. On ANV's Iris Xe profile under `exact_answers`.
#[test]
fn the_copy_of_a_snapshot_slot_refuses_a_form_its_fetch_s_lanes_do_not_hold() {
    let (Some(bc), Some(uint)) = (fixture("tob_renders"), assemble(UINT_READ)) else {
        return;
    };
    let stem = "intel-irisxe-anv";
    let desc = crate::pipeline::testing::description(stem).expect("description");
    let dev = Device::new(desc, DevicePolicy::default().exact_answers()).expect("device");
    // The float slot's fragment, and the unsigned integer slot's into an integer attachment.
    let legs = [
        ("uint", "sx_read_u", &uint, PixelFormat::RGBA8Uint, false),
        ("float", "tob_read", &bc, PixelFormat::RGBA8Unorm, true),
    ];
    for (leg, entry, module, attachment, float_lanes) in legs {
        let mut d = RenderPipelineDescriptor::new(function(&bc, "tob_rect"));
        d.fragment = Some(function(module, entry));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(attachment));
        let base = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{leg}: {e}"));
        let plan = base
            .plan(&dev, &state(None).shape())
            .unwrap_or_else(|e| panic!("{leg}: plan: {e}"));
        assert!(
            !plan.variant.snapshot_textures.is_empty(),
            "{leg}: no snapshot variant"
        );
        for form in FORMS {
            let mut key = plan.variant.clone();
            key.snapshot_textures = [(SLOT, form)].into();
            let r = dev.compile_render_variant(&d, &key, &Budget::standard());
            if holds(form, float_lanes) {
                r.unwrap_or_else(|e| panic!("{leg}/{form:?}: {e}"));
                continue;
            }
            let e = r.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(
                e.contains(&format!(
                    "the copy of snapshot slot 0 fetches lanes its form {form:?} does not hold"
                )),
                "{leg}/{form:?}: {e}"
            );
        }
    }
}

/// Each `OpImageFetch` of module `words`, with whether the image it fetches has an unsigned
/// integer sampled type: a slot's unsigned integer alias view, as against its own float image.
fn fetched_sampled_types(words: &[u32]) -> Vec<bool> {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::Op;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let word = |o: Option<&Operand>| match o {
        Some(Operand::Word(w) | Operand::IdRef(w)) => Some(*w),
        _ => None,
    };
    // The sampled type of each `OpTypeImage`, the result type of each value, and the integer
    // types: a declaration's own order puts each before its uses.
    let mut sampled = std::collections::HashMap::new();
    let mut types = std::collections::HashMap::new();
    let mut integers = std::collections::HashSet::new();
    let mut fetches = Vec::new();
    for i in &parsed.instructions {
        if let (Some(t), Some(id)) = (i.result_type, i.result_id) {
            types.insert(id, t);
        }
        match i.opcode {
            Op::TypeInt => integers.extend(i.result_id),
            Op::TypeImage => {
                if let (Some(id), Some(s)) = (i.result_id, word(i.operands.first())) {
                    sampled.insert(id, s);
                }
            }
            Op::ImageFetch => {
                let s = word(i.operands.first())
                    .and_then(|img| types.get(&img))
                    .and_then(|t| sampled.get(t));
                fetches.push(s.is_some_and(|s| integers.contains(s)));
            }
            _ => {}
        }
    }
    fetches
}

/// The copy of a snapshot whose form holds a 16-bit code fetches the slot's unsigned integer
/// alias under `exact_answers`, so the snapshot holds the texel's own 16 bits where the class
/// names 16-bit float channels and the read widens them (`lower::fragment`'s
/// `snapshot::fetches_half_bits`): the module fetches the slot twice, the copy's two exclusive
/// arms, one of them through an image of unsigned integer sampled type. The copy carries both
/// arms wherever the read does, the class word choosing between them in each pipeline.
///
/// On the fast default, where the slot takes no alias, and at a form of narrower codes, whose
/// texture's class names no such channels, the copy fetches the slot's own image alone: one
/// fetch, through its float image. The ledger's `half_texel_bits` row states which value each
/// policy answers.
#[test]
fn the_copy_of_a_half_float_snapshot_fetches_the_alias_under_exact_answers() {
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let mut d = RenderPipelineDescriptor::new(function(&bc, "tob_rect"));
    d.fragment = Some(function(&bc, "tob_read"));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    // A 16-bit float texture of each channel count, whose snapshot takes the arm, and the
    // normalized formats whose form is a narrower code's, whose copy fetches the image alone.
    // `RGBA32Float`'s form is the fetched lanes' bits, which hold a 16-bit code too, so its copy
    // carries the arm as well; its class names 32-bit float channels, so nothing takes it.
    let bound = [
        PixelFormat::R16Float,
        PixelFormat::RG16Float,
        PixelFormat::RGBA16Float,
        PixelFormat::RGBA8Unorm,
        PixelFormat::RGBA16Unorm,
        PixelFormat::RGBA32Float,
    ];
    let mut aliased = 0;
    for stem in PROFILES {
        for (label, policy) in [
            ("the fast default", DevicePolicy::default()),
            ("exact answers", DevicePolicy::default().exact_answers()),
        ] {
            let desc = crate::pipeline::testing::description(stem).expect("description");
            let dev = Device::new(desc, policy).expect("device");
            let base = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{stem}/{label}: {e}"));
            for pf in bound {
                let plan = base
                    .plan(&dev, &state(Some(pf)).shape())
                    .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: plan: {e}"));
                let form = plan
                    .variant
                    .snapshot_textures
                    .get(&SLOT)
                    .copied()
                    .unwrap_or_else(|| panic!("{stem}/{label}/{pf:?}: no snapshot form"));
                let variant = dev
                    .compile_render_variant(&d, &plan.variant, &Budget::standard())
                    .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: {e}"));
                let fetches: Vec<bool> = variant
                    .stages
                    .iter()
                    .flat_map(|m| fetched_sampled_types(&m.spirv))
                    .collect();
                // The copy carries the arm wherever the form holds a 16-bit code, as the read
                // carries it wherever the slot takes the alias: the class word, a specialization
                // constant, picks one of the two in each pipeline.
                let takes_alias = label == "exact answers"
                    && matches!(form, SnapshotForm::Half | SnapshotForm::Bits);
                let want: &[bool] = match takes_alias {
                    // The copy's arms: the slot's own image, and its unsigned integer alias.
                    true => &[false, true],
                    false => &[false],
                };
                let mut got = fetches.clone();
                got.sort_unstable();
                let mut want = want.to_vec();
                want.sort_unstable();
                assert_eq!(
                    got, want,
                    "{stem}/{label}/{pf:?} ({form:?}): fetched sampled types"
                );
                aliased += u32::from(takes_alias);
                for m in &variant.stages {
                    check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                        .unwrap_or_else(|e| panic!("{stem}/{label}/{pf:?}: L3: {e}"));
                }
            }
        }
    }
    assert_eq!(
        aliased,
        4 * PROFILES.len() as u32,
        "variants whose copy fetched the alias"
    );
}

/// Every snapshot form a texture whose class names 16-bit float channels of its own can take is
/// `SnapshotForm::Half` or `SnapshotForm::Bits`, the two the copy's alias arm stores the texel's
/// own 16 bits in: so the arm stands at every slot whose class can select it, and nowhere else
/// (`lower::texture`'s `half_bits_snapshot`).
///
/// The class word the binder states (`metal::texel_format::texel_format`) and the form it names
/// (`SnapshotForm::of` the slot's bound sampled view) are two readings of the one bound format,
/// so this pins them to each other over every `MTLPixelFormat` and every declared sampled type.
/// A swizzle only clears the class's bits (`TexelFormat::swizzled`), so it names no further
/// format.
#[test]
fn every_half_channel_format_names_a_snapshot_form_of_16_bit_codes() {
    use crate::device::requirements::FormatUsage;
    use crate::metal::texel_format::half_channels;
    use crate::pipeline::sym::SampledType;
    let usages: std::collections::BTreeSet<_> = [FormatUsage::SampledImage].into();
    let types = [
        SampledType::Float,
        SampledType::Sint,
        SampledType::Uint,
        SampledType::Uint64,
    ];
    let mut halves = 0;
    for stem in PROFILES {
        let desc = crate::pipeline::testing::description(stem).expect("description");
        let dev = Device::new(desc, DevicePolicy::default().exact_answers()).expect("device");
        for &pf in PixelFormat::ALL {
            if !half_channels(pf) {
                continue;
            }
            halves += 1;
            let Ok(m) = crate::metal::format::vk_format(pf, &usages, dev.profile()) else {
                continue;
            };
            for ty in types {
                let form = SnapshotForm::of(m.format, ty);
                assert!(
                    matches!(form, SnapshotForm::Half | SnapshotForm::Bits),
                    "{stem}/{pf:?}/{ty:?}: {form:?} is no form of the texel's own 16 bits"
                );
            }
        }
    }
    assert!(
        halves > 0,
        "a format whose class names 16-bit float channels"
    );
}

/// The entry's size estimate bounds the words every snapshot copy emits, the alias arm's among
/// them (`lower::fragment`'s `snapshot::SLOT_WORDS`): every form, on the fast default and under
/// `exact_answers`, where the copy has the second arm.
#[test]
fn the_snapshot_estimate_bounds_the_copy_it_emits() {
    use crate::library::function::FunctionConstantValues;
    use crate::pipeline::compile::{plan_bitcode, EntryRef};
    use crate::pipeline::lower_inputs::LoweringInputs;
    use crate::pipeline::testing::{compile_with, meter, profile};
    let Some(bc) = fixture("tob_renders") else {
        return;
    };
    let inputs = |form: SnapshotForm| {
        let mut inputs = LoweringInputs::new();
        inputs.snapshot_textures.insert(SLOT, form);
        inputs
    };
    let mut checked = 0;
    for stem in PROFILES {
        for policy in [
            DevicePolicy::default(),
            DevicePolicy::default().exact_answers(),
        ] {
            let p = profile(stem, policy).expect("profile");
            for form in FORMS {
                let inputs = inputs(form);
                let Ok(plan) = plan_bitcode(
                    &bc,
                    EntryRef::Name("tob_read"),
                    &FunctionConstantValues::defaults(),
                    &p,
                    &inputs,
                    &mut meter(),
                ) else {
                    continue;
                };
                let Ok(a) = compile_with(&bc, "tob_read", &p, &inputs) else {
                    continue;
                };
                let words = a.module.words.len() as u64;
                assert!(
                    plan.estimate.words >= words,
                    "{stem} {form:?} {:?}: estimate {} < emitted {words}",
                    policy.half_texel_bits,
                    plan.estimate.words
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "a profile that compiles a snapshot copy");
}
