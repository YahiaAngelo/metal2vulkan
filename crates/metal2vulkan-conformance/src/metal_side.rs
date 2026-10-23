//! The Metal side of the case vocabulary: native Metal through `objc2-metal`, on macOS only.
//!
//! This is what record mode runs. It exists so a case's Metal answer is taken by running the case
//! rather than copied off a terminal: the same case body, the same sites, the same transcript
//! shape.
//!
//! A shader reaches Metal as its fixture's typed-pointer AIR text: `xcrun metal-as` reassembles it
//! and `xcrun metallib` packages the library, which `newLibraryWithURL:` loads. Apple's assembler
//! reads that dialect and not the one a current `llvm-as` writes, which is why a fixture commits
//! both forms.
//!
//! Metal announces no degrade notice, so this side announces none: a notice in a transcript is
//! ours, and a case that holds one must name what justifies it.

use crate::case::{Answer, Failure, Handle, ShaderSource, Side, Venue};
use crate::transcript::Value;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};

/// The Metal level this side answers at: the host's own, since Metal answers as the release it runs
/// on.
const METAL_LEVEL: &str = crate::recording::ORACLE_OS_VERSION;

/// Native Metal, with the objects the case has made.
pub struct MetalSide {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    libraries: Vec<Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    command_buffers: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// The scratch directory the metallibs are built in, removed when the side is dropped.
    scratch: Scratch,
}

impl MetalSide {
    /// The system default device and a queue on it.
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_owned())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "MTLDevice::newCommandQueue returned nil".to_owned())?;
        Ok(Self {
            device,
            queue,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            command_buffers: Vec::new(),
            scratch: Scratch::new()?,
        })
    }

    /// The buffer `handle` names.
    fn held(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLBuffer>> {
        self.buffers
            .get(handle.0 as usize)
            .map(|buffer| &**buffer)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }
}

/// `kept` with `value` pushed on, answering its index.
fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// A directory the metallibs are built in.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "metal2vulkan-conformance-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `xcrun tool args`, answering why it could not.
fn xcrun(tool: &str, args: &[&std::ffi::OsStr]) -> Result<(), String> {
    let output = std::process::Command::new("/usr/bin/xcrun")
        .arg(tool)
        .args(args)
        .output()
        .map_err(|error| format!("xcrun {tool}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "xcrun {tool} failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

impl Side for MetalSide {
    fn venue(&self) -> Venue {
        Venue {
            device: self.device.name().to_string(),
            driver: "Metal".to_owned(),
            api: "Metal".to_owned(),
            metal_level: METAL_LEVEL.to_owned(),
        }
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let serial = self.libraries.len();
        let text = self.scratch.0.join(format!("{serial}.air.ll"));
        let air = self.scratch.0.join(format!("{serial}.air"));
        let metallib = self.scratch.0.join(format!("{serial}.metallib"));
        let build = || -> Result<(), String> {
            std::fs::write(&text, source.air_text)
                .map_err(|error| format!("{}: {error}", text.display()))?;
            xcrun(
                "metal-as",
                &[text.as_os_str(), "-o".as_ref(), air.as_os_str()],
            )?;
            xcrun(
                "metallib",
                &[air.as_os_str(), "-o".as_ref(), metallib.as_os_str()],
            )
        };
        build().map_err(|why| Failure::named("MetalToolFailed", why))?;
        let path = metallib
            .to_str()
            .ok_or_else(|| Failure::named("NonUtf8Path", metallib.display().to_string()))?;
        let library = self
            .device
            .newLibraryWithURL_error(&NSURL::fileURLWithPath(&NSString::from_str(path)))
            .map_err(|error| Failure::named("NewLibraryWithURL", error.to_string()))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let buffer = self.held(handle)?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes of shared storage,
        // and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let buffer = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        buffers: &[Handle],
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the kernel's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, at) };
        }
        let size = |axes: [u32; 3]| MTLSize {
            width: axes[0] as usize,
            height: axes[1] as usize,
            depth: axes[2] as usize,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(size(threadgroups), size(threads));
        encoder.endEncoding();
        command_buffer.commit();
        Ok(keep(&mut self.command_buffers, command_buffer))
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let command_buffer = self
            .command_buffers
            .get(command_buffer.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("command buffer {}", command_buffer.0),
                )
            })?;
        command_buffer.waitUntilCompleted();
        Ok(Value::unit_variant(status_word(command_buffer.status())))
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = offset
            .checked_add(length as u64)
            .is_none_or(|end| end > buffer.length() as u64);
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        // SAFETY: `[offset, offset + length)` is inside the buffer's shared storage, and the
        // caller waited for the command buffer that wrote it.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                buffer.contents().as_ptr().cast::<u8>().add(offset as usize),
                length,
            )
        };
        Ok(bytes.to_vec())
    }

    fn announced(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// The word a command buffer's status records as, in the runtime's spelling so the two sides'
/// transcripts hold one vocabulary.
fn status_word(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        _ => "Unknown",
    }
}
