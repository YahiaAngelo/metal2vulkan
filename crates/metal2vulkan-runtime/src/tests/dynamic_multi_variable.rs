//! A set-0 buffer binding an entry point uses through more than one variable: whether the device
//! takes it as a dynamic descriptor, and that its answer agrees with what its pipelines do.
//!
//! A slot or store sink declares one variable per view at its binding -- a word view, and the
//! narrow or `uvec4` unit views an access needs -- so a kernel whose dropped stores move a narrow
//! leaf has a binding of several variables (`pipeline::objects::dynamic_bindings`). Vulkan bounds
//! the variables at a binding nowhere, but MoltenVK's SPIRV-Cross writes Metal that does not
//! compile for a **dynamic** one, so the device is asked (one pipeline of that shape created and
//! destroyed, `objects::takes_multi_variable_dynamic`) rather than every device paying for it.
//!
//! What each test holds, on any device:
//!
//! - the pipelines of such a kernel are made and run, whatever the device answered, and
//!   `record_stride_not_a_multiple_of_four` answers exactly what Metal's `sizeof` and its two
//!   narrow stores say;
//! - and the binding type follows the answer: where the device takes it and the variant does not
//!   push set 0, every dispatch after the first binds its predecessor's set, and where it does
//!   not, each allocates and writes its own.
//!
//! On an Apple M3 Max under MoltenVK 1.4.2 the answer is no, so these hold the second branch
//! there; a driver that takes it holds the first.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use crate::{
    Device, DeviceKind, DeviceOptions, PipelineOptions, ResourceOptions, ShaderCacheOptions, Size,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "m2v-dynmulti-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// What a run of `entry` answered: the bytes of each bound buffer, and the counter steps.
struct Ran {
    buffers: Vec<Vec<u8>>,
    reused: u64,
    found: u64,
    pushed: u64,
}

/// Dispatch `entry` of fixture `file` `dispatches` times with `slots` buffers of `bytes` bytes
/// bound, each its own buffer, `threads` threads a dispatch.
fn run(
    file: &str,
    entry: &str,
    slots: u32,
    bytes: u64,
    threads: u64,
    dispatches: usize,
) -> Result<Ran, String> {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(file);
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).map_err(|e| format!("{e:?}"))?;
    let library = metal2vulkan::library::Library::from_air(&bitcode).map_err(|e| format!("{e}"))?;
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(entry)
        .map_err(|e| format!("{e}"))?;
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .map_err(|e| format!("pipeline: {e}"))?;
    let buffers: Vec<_> = (0..slots)
        .map(|_| device.new_buffer(bytes, ResourceOptions::shared()).unwrap())
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
    }
    let before = device.work_counts();
    for _ in 0..dispatches {
        encoder
            .dispatch_threads(Size::new(threads, 1, 1), Size::new(threads.min(32), 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(std::time::Duration::from_secs(60))
        .unwrap();
    if let Some(e) = command_buffer.error() {
        return Err(format!("command buffer: {e:?}"));
    }
    let after = device.work_counts();
    Ok(Ran {
        buffers: buffers
            .iter()
            .map(|b| {
                let p = b.contents().unwrap();
                // SAFETY: the command buffer completed, so the GPU's writes are visible, and the
                // mapping holds the buffer's whole length.
                unsafe { std::slice::from_raw_parts(p.as_ptr(), b.length() as usize) }.to_vec()
            })
            .collect(),
        reused: after.set_zero_sets_reused - before.set_zero_sets_reused,
        found: after.set_zero_sets_found - before.set_zero_sets_found,
        pushed: after.dispatches_pushing_set_zero - before.dispatches_pushing_set_zero,
    })
}

/// `record_stride_not_a_multiple_of_four`: `sizeof(Rec)` is 18 and `alignof` 2, so
/// `sizes[tid] == 18002`, and its two narrow stores land at `18 * tid + 6` (a `short`) and
/// `18 * tid + 11` (a `uchar`). Its store sink uses a `u16` and a `u8` unit view, so its sink
/// binding is one of three variables.
#[test]
fn a_narrow_unit_kernel_runs_and_its_set_follows_the_devices_answer() {
    const THREADS: u64 = 8;
    let answer = answered();
    for dispatches in [8usize, 16] {
        let ran = run(
            "kernel_record_stride_not_a_multiple_of_four.ll",
            "record_stride_not_a_multiple_of_four",
            2,
            1024,
            THREADS,
            dispatches,
        )
        .unwrap_or_else(|e| panic!("{dispatches} dispatches: {e}"));
        let recs = &ran.buffers[0];
        let sizes = &ran.buffers[1];
        for tid in 0..THREADS as usize {
            let size = u32::from_le_bytes(sizes[tid * 4..tid * 4 + 4].try_into().unwrap());
            let short = i16::from_le_bytes(recs[18 * tid + 6..18 * tid + 8].try_into().unwrap());
            let byte = recs[18 * tid + 11];
            assert_eq!(
                (size, short, byte),
                (18002, (tid as i16) * 7 + 1, (tid as u8) ^ 0xa5),
                "thread {tid} of {dispatches} dispatches answered ({size}, {short}, {byte}): \
                 Metal's `sizeof` is 18 and `alignof` 2, and the two narrow stores land at \
                 18 * tid + 6 and 18 * tid + 11"
            );
        }
        ran.holds_the_answer(answer, dispatches, "record_stride_not_a_multiple_of_four");
    }
}

/// `encoders_plain` of the reflection fixtures, whose store sink uses a `u16`, a `u8` and a
/// `uvec4` unit view: its sink binding is one of four variables. Its answers are not Metal's to
/// check here (its buffers are its own), only that it is made, runs, answers the same bytes at
/// both dispatch counts, and takes the binding type the device's answer calls for.
#[test]
fn a_four_variable_kernel_runs_and_its_set_follows_the_devices_answer() {
    let answer = answered();
    let mut digests = Vec::new();
    for dispatches in [8usize, 16] {
        let ran = run(
            "reflection/encoders.ll",
            "encoders_plain",
            8,
            4096,
            4,
            dispatches,
        )
        .unwrap_or_else(|e| panic!("{dispatches} dispatches: {e}"));
        digests.push(ran.buffers.clone());
        ran.holds_the_answer(answer, dispatches, "encoders_plain");
    }
    assert_eq!(
        digests[0], digests[1],
        "eight and sixteen dispatches of `encoders_plain` answered different bytes"
    );
}

/// The device's own answer, from a device of this test's own making: what every check here reads.
fn answered() -> bool {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    crate::pipeline::objects::asks_multi_variable_dynamic(&device)
}

impl Ran {
    /// Hold the binding type the device's `answer` calls for: where it takes a binding of several
    /// variables dynamically and the variant does not push set 0, every dispatch after the first
    /// binds the set its predecessor wrote; where it does not, each allocates and writes its own,
    /// which `set_zero_sets_reused` and `set_zero_sets_found` both count none of.
    fn holds_the_answer(&self, answer: bool, dispatches: usize, what: &str) {
        if self.pushed > 0 {
            assert_eq!(
                self.pushed, dispatches as u64,
                "{what}: {} of {dispatches} dispatches pushed set 0, which is all or none",
                self.pushed
            );
            return;
        }
        let want = match answer {
            true => dispatches as u64 - 1,
            false => 0,
        };
        assert_eq!(
            (self.reused, self.found),
            (want, 0),
            "{what}: the device {} a binding of several variables dynamically, and {dispatches} \
             dispatches reused {} and found {} of their predecessors' sets, not {want} and 0",
            match answer {
                true => "takes",
                false => "does not take",
            },
            self.reused,
            self.found
        );
    }
}
