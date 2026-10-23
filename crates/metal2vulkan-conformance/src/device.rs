//! Opening a runtime device for the suite, and collecting the notices it announces.
//!
//! The device is made with explicit options and no environment: exact answers, a shader cache
//! directory this run owns and removes, and a message sink that keeps the degrade-notice codes the
//! device delivered. A warm cache never stands in for a translation the suite compares, and a
//! notice is never lost: it is the thing that justifies a loosened field.

use metal2vulkan_runtime::{
    DebugMessage, Degradation, Device, DeviceKind, DeviceOptions, MessageSink, ShaderCacheOptions,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The degrade-notice codes a device announced, which [`Notices::take`] hands over and clears.
#[derive(Clone, Default)]
pub struct Notices(Arc<Mutex<Vec<String>>>);

impl Notices {
    /// A sink for a device's options: it keeps each notice's code and drops the rest.
    pub fn sink(&self) -> MessageSink {
        let kept = Arc::clone(&self.0);
        MessageSink::new(move |message: &DebugMessage| {
            if let Some(notice) = message.degradation.as_ref() {
                if let Ok(mut codes) = kept.lock() {
                    codes.push(notice.degradation.code().to_owned());
                }
            }
        })
    }

    /// The codes announced so far, taken away.
    pub fn take(&self) -> Vec<String> {
        self.0
            .lock()
            .map(|mut codes| std::mem::take(&mut *codes))
            .unwrap_or_default()
    }
}

/// A shader cache directory one run owns: a new, empty directory under the OS temp directory,
/// removed when dropped, after the device that used it.
pub struct ScratchCache(PathBuf);

impl ScratchCache {
    /// A directory this run owns, not yet made (the device makes it).
    pub fn new() -> std::io::Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        Ok(Self(std::env::temp_dir().join(format!(
            "m2v-conformance-shader-cache-{}-{nanos}",
            std::process::id()
        ))))
    }

    /// The directory.
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for ScratchCache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Which physical devices to open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Venues {
    /// A GPU: discrete, integrated or virtual.
    Gpu,
    /// A CPU implementation (lavapipe), which the no-Metal replay leg runs on.
    Cpu,
}

impl Venues {
    /// The device kinds it names.
    fn kinds(self) -> Vec<DeviceKind> {
        match self {
            Venues::Gpu => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
            Venues::Cpu => vec![DeviceKind::Cpu],
        }
    }
}

/// The Vulkan loader to open, where this host does not have one on its default path.
fn loader() -> Option<PathBuf> {
    // SAFETY: opening the loader library runs only its initializers; the entry is dropped.
    if unsafe { ash::Entry::load() }.is_ok() {
        return None;
    }
    [
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.dylib",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

/// The suite's device options: exact answers, every notice at every occurrence, the cache this
/// run owns, and `notices`' sink.
///
/// `announce_each` over every kind is what makes a case's notices its own. A device announces a
/// kind once by default, so on a shared device the first case to provoke a notice holds it and
/// every later case sees nothing: the same case then reads as announced or as exact depending on
/// which cases ran before it in the same process. That would make the notices site — the suite's
/// one mechanism for "a difference is a bug unless we announced it" — depend on the selection and
/// its order. At every occurrence, each case's own work announces its own notices, so a case's
/// verdict is the case's.
pub fn options(notices: &Notices, cache: &ScratchCache) -> DeviceOptions {
    let mut options = DeviceOptions::default();
    options.policy = options.policy.exact_answers();
    options.policy.sampling = metal2vulkan::device::SamplingPolicy::Exact;
    options.announce_each = Degradation::ALL.iter().copied().collect();
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    options.messages = Some(notices.sink());
    options
}

/// The runtime's default device among the physical devices `venues` names.
pub fn open(venues: Venues, notices: &Notices, cache: &ScratchCache) -> Result<Device, String> {
    let mut options = options(notices, cache);
    options.kinds = venues.kinds();
    Device::system_default_with(&options)
        .map_err(|error| format!("the runtime made no {venues:?} device: {error}"))
}
