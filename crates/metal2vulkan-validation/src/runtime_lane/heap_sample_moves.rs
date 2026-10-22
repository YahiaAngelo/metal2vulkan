//! A heap's large multisample texture's moves between its image and Metal's tiles
//! (`texture/heap_tiled.rs`): the images they go through, what the process holds for them, and
//! what a second texture of the same shape adds.

use super::host_memory::run;
use super::{validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, CpuCacheMode, Device, Error, HazardTrackingMode,
    HeapDescriptor, HeapType, Origin, ResourceOptions, Size, StorageMode, Texture,
    TextureDescriptor,
};

/// The process's physical footprint and its peak so far, in MiB, where the OS answers them
/// (`proc_pid_rusage`, macOS); `None` elsewhere.
fn footprint() -> Option<(u64, u64)> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `rusage_info_v4` is plain integers, for which zeros are a value.
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let pid = i32::try_from(std::process::id()).ok()?;
        let at = (&mut info as *mut libc::rusage_info_v4).cast::<libc::rusage_info_t>();
        // SAFETY: the flavor names the struct `at` points at, which the call fills and nothing
        // else.
        let rc = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, at) };
        (rc == 0).then_some((
            info.ri_phys_footprint >> 20,
            info.ri_lifetime_max_phys_footprint >> 20,
        ))
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// A blit of a 16 x 16 square of each of `texture`'s aspects from a buffer, at `(x, 0)` of its
/// slice 0: the texture's shadow copied in before it and back after it.
fn blit_into(queue: &CommandQueue, texture: &Texture, buffer: &Buffer, x: u64) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (unit, option) in [
        (4, BlitOption::DepthFromDepthStencil),
        (1, BlitOption::StencilFromDepthStencil),
    ] {
        let row = 16 * 4 * unit;
        blit.copy_from_buffer_to_texture(
            buffer,
            0,
            row,
            row * 16,
            Size::new(16, 16, 1),
            texture,
            0,
            0,
            Origin::new(x, 0, 0),
            option,
        )
        .unwrap();
    }
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// A 4096 x 2160 `Depth32Float_Stencil8` texture of 4 samples (`allowGPUOptimizedContents` off,
/// which Metal keeps uncompressed) in a `Shared` placement heap, a shadow of its samples in
/// Metal's tiles there where the device moves them (`WorkCounts::heap_textures_tiled`), blitted
/// into twice, then a second one: the moves' images are made as the first blit's copies are
/// recorded, none as a texture is made, and every later copy, the second texture's too, takes
/// those its queue kept and makes none (`WorkCounts::heap_sample_move_images_made` and
/// `heap_sample_move_images_lent`, `texture/copy_images.rs`). `MOVES` lines print the process's
/// footprint at each step and its peak.
#[test]
fn a_large_heap_multisample_texture_s_moves() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device: &Device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let options = ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    );
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, 4096, 2160, false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = 4;
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options;
    descriptor.allow_gpu_optimized_contents = false;
    let answer = match device.heap_texture_size_and_align(&descriptor) {
        Err(Error::Unsupported(u)) if u.feature == Feature::TextureSampleCount => {
            eprintln!("MOVES: the device takes no such sample count");
            return;
        }
        answer => answer.unwrap(),
    };
    let samples = device.new_buffer(16 * 16 * 4 * 4, options).unwrap();
    let print = |step: &str| match footprint() {
        Some((now, peak)) => eprintln!("MOVES {step}: footprint {now} MiB, peak {peak} MiB"),
        None => eprintln!("MOVES {step}"),
    };
    print("start");
    let moved = || {
        let counts = device.work_counts();
        (
            counts.heap_sample_move_images_made,
            counts.heap_sample_move_images_lent,
        )
    };
    let mut kept = Vec::new();
    for second in [false, true] {
        let what = if second { "second" } else { "first" };
        let heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Placement,
                answer.size,
                options,
            ))
            .unwrap();
        print(&format!("{what} heap made ({} MiB)", answer.size >> 20));
        let before = device.work_counts().heap_textures_tiled;
        let images = moved();
        let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let tiled = device.work_counts().heap_textures_tiled - before == 1;
        print(&format!("{what} texture made, tiled {tiled}"));
        if !tiled {
            return;
        }
        assert_eq!(moved(), images, "{what}: images made with the texture");
        for round in 0..2 {
            let (made, lent) = moved();
            blit_into(&queue, &texture, &samples, 16 * round);
            let (now_made, now_lent) = moved();
            let (made, lent) = (now_made - made, now_lent - lent);
            print(&format!(
                "{what} texture blitted into, round {round}: {made} images made, {lent} lent"
            ));
            match (second, round) {
                (false, 0) => assert!(made > 0, "{what}, round {round}: no image made"),
                _ => assert_eq!(made, 0, "{what}, round {round}: images made again"),
            }
            assert!(lent > 0, "{what}, round {round}: no image lent");
        }
        kept.push((heap, texture));
    }
    drop(kept);
    print("both gone");
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}
