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

/// Which of the two forms of every speed-for-accuracy choice the device takes.
///
/// Both are the suite's business. `Exact` is what a conformance run should hold the runtime to;
/// `Default` is what a consumer gets, and comparing a case under both is how the suite says
/// whether a difference is the policy's or the implementation's. The owner's rule keeps both forms
/// and asks for tests of both settings, so the command can select either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// `DevicePolicy::exact_answers()`, with exact sampling: the accurate form of every choice.
    Exact,
    /// The runtime's own defaults: the fast form of every choice, which is what a consumer runs.
    Default,
}

/// The suite's device options under `policy`: every notice at every occurrence, the cache this run
/// owns, and `notices`' sink.
///
/// `announce_each` over every kind is what makes a case's notices its own. A device announces a
/// kind once by default, so on a shared device the first case to provoke a notice holds it and
/// every later case sees nothing: the same case then reads as announced or as exact depending on
/// which cases ran before it in the same process. That would make the notices site — the suite's
/// one mechanism for "a difference is a bug unless we announced it" — depend on the selection and
/// its order. At every occurrence, each case's own work announces its own notices, so a case's
/// verdict is the case's, under either policy.
pub fn options_under(policy: Policy, notices: &Notices, cache: &ScratchCache) -> DeviceOptions {
    let mut options = DeviceOptions::default();
    if policy == Policy::Exact {
        options.policy = options.policy.exact_answers();
        options.policy.sampling = metal2vulkan::device::SamplingPolicy::Exact;
    }
    options.announce_each = Degradation::ALL.iter().copied().collect();
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    options.messages = Some(notices.sink());
    options
}

/// The suite's device options: exact answers, the cache this run owns, and `notices`' sink.
pub fn options(notices: &Notices, cache: &ScratchCache) -> DeviceOptions {
    options_under(Policy::Exact, notices, cache)
}

/// The runtime's default device among the physical devices `venues` names, under `policy`.
pub fn open_under(
    venues: Venues,
    policy: Policy,
    notices: &Notices,
    cache: &ScratchCache,
) -> Result<Device, String> {
    let mut options = options_under(policy, notices, cache);
    options.kinds = venues.kinds();
    Device::system_default_with(&options)
        .map_err(|error| format!("the runtime made no {venues:?} device: {error}"))
}

/// The runtime's default device among the physical devices `venues` names, with exact answers.
pub fn open(venues: Venues, notices: &Notices, cache: &ScratchCache) -> Result<Device, String> {
    open_under(venues, Policy::Exact, notices, cache)
}

/// The guard a family of **pure-query cases** rests on: run every case of a selection on the
/// runtime, and name every site whose answer is the runtime's `Error::Invalid`.
///
/// `Error::Invalid` is this tree's one reading of "Metal's own validation stops this call, which
/// ends Metal's process" — [`crate::cases::heap_metal_sizes`] and [`crate::cases::sparse_heaps`]
/// state the distinction and each hold their own tables to it by hand. This is the same guard,
/// generalized to the whole transcript: a family whose cases carry on past a refusal records it at
/// a site instead of aborting, so the thing to scan is the tree rather than the return value.
///
/// It is also the **order of work** those two modules describe: a table is replayed on the runtime
/// first, where an `Invalid` is a recorded value and nothing else, and only a table that answers no
/// `Invalid` anywhere is ever handed to Metal.
///
/// `Err` is the reason no verdict could be taken at all — no device here — which a caller reports
/// rather than passing over. Every other refusal (ours, or Metal's own nil) is a legitimate
/// recorded value and is not named.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub fn sites_the_runtime_calls_invalid(
    cases: &[Box<dyn crate::case::Case>],
) -> Result<Vec<String>, String> {
    use crate::transcript::{Node, Value};

    /// Whether a recorded value is the runtime's `Error::Invalid`. `Failure` keeps the error as the
    /// variant path a transcript holds, not as the typed error, so the kind is read from there.
    fn is_invalid(value: &Value) -> bool {
        let Value::Variant { name, payload } = value else {
            return false;
        };
        if name != "Err" {
            return false;
        }
        matches!(&**payload, Value::Variant { name, .. } if name == "Error::Invalid")
    }

    /// Every site under `node` whose answer is `Invalid`, by its path.
    fn walk(path: &str, node: &Node, found: &mut Vec<String>) {
        let here = match path.is_empty() {
            true => node.name.to_string(),
            false => format!("{path}/{}", node.name),
        };
        if is_invalid(&node.observed.value) {
            found.push(here.clone());
        }
        for child in &node.children {
            walk(&here, child, found);
        }
    }

    let notices = Notices::default();
    let cache = ScratchCache::new().map_err(|why| format!("no scratch cache: {why}"))?;
    let venue = if cfg!(m2v_gpu) {
        Venues::Gpu
    } else {
        Venues::Cpu
    };
    let runtime = open(venue, &notices, &cache)?;
    let mut found = Vec::new();
    for case in cases {
        let taker = notices.clone();
        let mut side = crate::runtime_side::RuntimeSide::new(&runtime, move || taker.take());
        let mut sites = crate::transcript::Sites::new();
        // An abort is the case's own refusal of a call it cannot carry on without, which is a
        // recorded verdict and not a validation stop; what it leaves emitted is still scanned.
        let aborted = case.run(&mut side, &mut sites).err();
        // `finish` consumes the sites, which is why the abort is kept first.
        let mut here = Vec::new();
        walk("", &sites.finish().root, &mut here);
        for site in here {
            found.push(format!("{}: {site}", case.name()));
        }
        if let Some(abort) = aborted {
            if is_invalid(&abort.failure.recorded()) {
                found.push(format!("{}: the call that aborted the case", case.name()));
            }
        }
    }
    Ok(found)
}
