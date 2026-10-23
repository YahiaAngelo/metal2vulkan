//! The case API: a case is written once and run by either side.
//!
//! A case names itself, sets its resources up, runs, and emits the sites it observes
//! ([`Case::run`]). It is written against [`Side`], the vocabulary both sides answer in, so porting
//! a case is a matter of saying which calls it makes and which sites it emits, and nothing about
//! which implementation is running it.
//!
//! A side with no public call for something a case reads answers [`Failure::unavailable`] rather
//! than pretending: Metal exposes an indirect command's read-back only privately, so the Metal side
//! answers `Unavailable` there and the case's Metal answer is carried from a measurement this tree
//! already holds ([`MetalSource::Carried`]). The case body stays one body.
//!
//! The vocabulary grows one family at a time as cases are ported. Nothing here reads an
//! environment variable, a device name or a vendor: a case that needs a feature finds out by making
//! the call and reading the typed refusal.

use crate::allowance::Allowance;
use crate::recording::ShaderRef;
use crate::transcript::{Node, Observed, Sites, Transcript, Value};
use crate::verdict::device_gap;
use metal2vulkan::metal::{AttributeFormat, IndexType, PixelFormat, StepFunction};

/// A shader as both sides can take it.
///
/// One shader, two committed texts, because the two sides read different dialects of it: the
/// runtime parses the bitcode `llvm-as` assembles from the fixture's `.ll`, and Metal's own
/// assembler reads the fixture's `.air.ll` (typed pointers), which `xcrun metal-as` reassembles and
/// `xcrun metallib` packages. A case's input fingerprint covers both, so a change to either is a
/// change to the case's inputs.
#[derive(Clone, Copy, Debug)]
pub struct ShaderSource<'a> {
    /// The AIR bitcode the runtime parses.
    pub bitcode: &'a [u8],
    /// The fixture's typed-pointer AIR text, which Metal's assembler reads.
    pub air_text: &'a str,
    /// The entry point, for a side that names it as it makes the library.
    pub entry: &'a str,
}

/// A resource a case made on a side. The number is the side's own; nothing compares it, and no
/// recording holds one, because a Metal object's identity is not observable across hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handle(pub u32);

/// What a dispatch binds: the buffers at 0 up, and the textures at 0 up.
///
/// One value rather than one argument per kind, so a case that binds a kind the vocabulary gains
/// later does not change every call site, and so a reader of a case sees the whole argument table
/// in one place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bindings<'a> {
    /// The buffers, bound at Metal buffer index 0 up in this order.
    pub buffers: &'a [Handle],
    /// The textures, bound at Metal texture index 0 up in this order.
    pub textures: &'a [Handle],
}

impl<'a> Bindings<'a> {
    /// `buffers` at 0 up, and no texture.
    pub const fn buffers(buffers: &'a [Handle]) -> Self {
        Self {
            buffers,
            textures: &[],
        }
    }

    /// `textures` at 0 up, and no buffer.
    pub const fn textures(textures: &'a [Handle]) -> Self {
        Self {
            buffers: &[],
            textures,
        }
    }
}

/// A kernel's `[[stage_in]]` descriptor, as both sides take it: Metal's
/// `MTLStageInputOutputDescriptor`.
///
/// The vocabulary is Metal's own enumerations, as the translator's public API names them, so a
/// value a case writes reaches `MTLAttributeFormat`, `MTLStepFunction` and `MTLIndexType` by their
/// raw ordinals and reaches the runtime's descriptor unchanged. There is no second spelling to
/// keep in step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageInput {
    /// The attributes a kernel reads, by their Metal attribute index.
    pub attributes: Vec<StageInAttribute>,
    /// The buffer layouts the attributes read, by their Metal buffer index.
    pub layouts: Vec<StageInLayout>,
    /// The index type an indexed step reads.
    pub index_type: IndexType,
    /// The Metal buffer index an indexed step reads its indices from.
    pub index_buffer_index: u32,
}

impl StageInput {
    /// A descriptor of `attributes` and `layouts`, whose indices are 32-bit words at
    /// `index_buffer_index`.
    pub fn new(
        attributes: Vec<StageInAttribute>,
        layouts: Vec<StageInLayout>,
        index_buffer_index: u32,
    ) -> Self {
        Self {
            attributes,
            layouts,
            index_type: IndexType::UInt32,
            index_buffer_index,
        }
    }
}

/// One `[[stage_in]]` attribute: Metal's `MTLAttributeDescriptor` at an attribute index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInAttribute {
    /// The Metal attribute index, as the shader declares it.
    pub index: u32,
    pub format: AttributeFormat,
    /// The attribute's byte offset inside an element.
    pub offset: u32,
    /// The Metal buffer index of the layout it reads.
    pub buffer_index: u32,
}

/// One `[[stage_in]]` buffer layout: Metal's `MTLBufferLayoutDescriptor` at a buffer index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageInLayout {
    /// The Metal buffer index the layout is of.
    pub buffer_index: u32,
    /// Bytes between elements; `None` is `MTLBufferLayoutStrideDynamic`, the stride the encoder
    /// binds the buffer with.
    pub stride: Option<u32>,
    pub step_function: StepFunction,
    /// The step rate; Metal requires 0 of a constant step.
    pub step_rate: u32,
}

/// The stage-in region a dispatch fetches its `[[stage_in]]` attributes from: the two calls Metal
/// sets it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageInRegion {
    /// `setStageInRegion:`.
    Direct { origin: [u32; 3], size: [u32; 3] },
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`, whose buffer holds the origin's
    /// three words and then the size's.
    Indirect { buffer: Handle, offset: u64 },
}

/// What a side answered a call with.
pub type Answer<T> = Result<T, Failure>;

/// A call that did not answer: the typed error as the transcript records it, and whether it is the
/// venue's honest gap rather than ours.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The error, as a `Value::error` of the variant's path and its payload.
    pub value: Value,
    /// What the venue lacks, where the typed refusal names capabilities a device could enable or
    /// a measured driver defect ([`device_gap`]). `None` where the refusal is ours to answer for.
    pub lacks: Option<String>,
}

impl Failure {
    /// The failure a typed runtime error is.
    pub fn of(error: &metal2vulkan_runtime::Error) -> Self {
        Self {
            value: Value::error(error_variant(error), Value::Lines(vec![error.to_string()])),
            lacks: device_gap(error),
        }
    }

    /// The failure a side with no public call for `call` answers. It is neither a divergence nor a
    /// refusal: it says the side cannot be asked, and a case whose Metal side answers it carries
    /// its Metal answer instead.
    pub fn unavailable(call: &str) -> Self {
        Self {
            value: Value::error(
                "Unavailable",
                Value::fields([("call", Value::Lines(vec![call.to_owned()]))]),
            ),
            lacks: None,
        }
    }

    /// A failure of `variant` with `detail`, for a side whose call failed outside the runtime's
    /// typed errors.
    pub fn named(variant: &str, detail: impl Into<String>) -> Self {
        Self {
            value: Value::error(variant, Value::Lines(vec![detail.into()])),
            lacks: None,
        }
    }
}

/// The variant path a typed runtime error records as: the error's outermost variant, which is what
/// a consumer matches on.
fn error_variant(error: &metal2vulkan_runtime::Error) -> String {
    let debug = format!("{error:?}");
    let head = debug
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .find(|word| !word.is_empty())
        .unwrap_or("Error");
    format!("Error::{head}")
}

/// Where a case's run stopped: a call it needed did not answer. An abort is not a verdict; the
/// comparison reads the sites emitted up to it, and the refused call is one of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Abort {
    /// The site the call was emitted at.
    pub at: String,
    pub failure: Failure,
}

/// The venue a side runs on: what a result carries so a report can say where it ran. A case never
/// branches on any of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Venue {
    /// The physical device's name.
    pub device: String,
    /// The driver, as the device reports it.
    pub driver: String,
    /// The API version the device runs.
    pub api: String,
    /// The Metal level the side answers as ([`metal2vulkan_runtime::MetalLevel::macos`]).
    pub metal_level: String,
}

impl Venue {
    /// The short label the report's matrix prints as a column.
    pub fn label(&self) -> String {
        self.device.clone()
    }
}

impl std::fmt::Display for Venue {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{} ({}, API {}, Metal level {})",
            self.device, self.driver, self.api, self.metal_level
        )
    }
}

/// The vocabulary a case is written against.
///
/// Every call answers [`Answer`], so a case records what it got and carries on or stops. A side
/// that has no public call for one answers [`Failure::unavailable`].
pub trait Side {
    /// Where the side runs.
    fn venue(&self) -> Venue;

    /// A library of the shader's AIR, from whichever of its forms the side reads
    /// ([`ShaderSource`]).
    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle>;

    /// `newComputePipelineStateWithFunction:error:` of the library's `entry`.
    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle>;

    /// `newBufferWithBytes:length:options:` of shared storage.
    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle>;

    /// `newBufferWithLength:options:` of shared storage, zeroed.
    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle>;

    /// A 2D texture of `format`, `size` texels, one level, shader-read and shader-write, of shared
    /// storage, whose level 0 is then `bytes` at `row_pitch` bytes a row
    /// (`replaceRegion:mipmapLevel:withBytes:bytesPerRow:`).
    ///
    /// Every case states its own row pitch, because that pitch is part of what the case means and
    /// a side that computed its own could disagree with the other silently.
    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle>;

    /// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` of a texture's whole level 0, at `row_pitch`
    /// bytes a row.
    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>>;

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up per
    /// kind, one `dispatchThreadgroups:threadsPerThreadgroup:`, and `commit`.
    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle>;

    /// `waitUntilCompleted` and then `status`, as a variant of the status's name.
    fn wait(&mut self, command_buffer: Handle) -> Answer<Value>;

    /// `contents` of a shared buffer: `length` bytes from `offset`.
    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>>;

    /// The degrade notices the side has announced so far, by code, and taken away. Metal announces
    /// none, so its side answers an empty list.
    fn announced(&mut self) -> Vec<String>;

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` of the library's `entry`,
    /// under the stage-input descriptor `stage_input` ([`StageInput`]).
    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle>;

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up per
    /// kind, each of `stage_in` set in the order the case lists them, one
    /// `dispatchThreads:threadsPerThreadgroup:` of `grid` threads, and `commit`.
    ///
    /// `stage_in` is a list rather than one region because setting it twice is a case of its own:
    /// Metal's later call replaces the earlier, and a case states that by listing both.
    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle>;

    /// The read-back of an indirect render command the CPU encoded, as the side's getters answer
    /// it: `getCommandType` and the draw. Metal exposes these privately only, so the Metal side
    /// answers [`Failure::unavailable`] and such a case's Metal answer is carried.
    ///
    /// `primitive_type` is the raw `MTLPrimitiveType` value, so a case can set one of Metal's
    /// private types; `indexed` chooses the indexed setter.
    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        let _ = (primitive_type, indexed);
        Err(Failure::unavailable("indirect_draw_read_back"))
    }

    /// Run a copy-and-fill program ([`crate::work`]) and answer what it read.
    ///
    /// Metal fills a texture through private API and its two-plane extended-range formats are
    /// private too, so the Metal side has no public call for this and answers
    /// [`Failure::unavailable`]; a case built on it carries its Metal answer.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        let _ = work;
        Err(Failure::unavailable("work"))
    }
}

/// What an indirect render command's getters answered, beside what the whole record holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndirectReadBack {
    /// `getCommandType`'s answer, as its bits.
    pub command_type: u64,
    /// The primitive type the draw getter answered, or `None` where it answered no draw.
    pub answered_primitive_type: Option<u64>,
    /// The primitive type the whole record's read-back keeps.
    pub recorded_primitive_type: Option<u64>,
}

/// Where a case's Metal answer comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalSource {
    /// Record mode runs the case on Metal.
    Run,
    /// Metal has no public call for what the case reads, or running it on Metal is not safe, so
    /// its answer is carried from a measurement this tree already holds: the module whose table it
    /// came from, and whether record mode may replace it.
    Carried {
        from: &'static str,
        re_record: ReRecord,
    },
}

impl MetalSource {
    /// A carried answer record mode may replace by running the case on Metal.
    pub const fn carried(from: &'static str) -> Self {
        Self::Carried {
            from,
            re_record: ReRecord::Allowed,
        }
    }

    /// A carried answer record mode must refuse to replace, because running the case on Metal is
    /// not safe.
    pub const fn carried_only(from: &'static str, reason: &'static str) -> Self {
        Self::Carried {
            from,
            re_record: ReRecord::Forbidden { reason },
        }
    }

    /// Why record mode must not run the case on Metal, where it must not.
    pub const fn forbidden(&self) -> Option<&'static str> {
        match self {
            MetalSource::Carried {
                re_record: ReRecord::Forbidden { reason },
                ..
            } => Some(reason),
            MetalSource::Run | MetalSource::Carried { .. } => None,
        }
    }
}

/// Whether record mode may run a case on Metal.
///
/// A case whose Metal leg faults the GPU, trips its watchdog or ends the process carries its
/// answer and forbids re-recording. Record mode refuses such a case by this flag, so a later
/// re-record of a whole selection cannot crash the GPU or the process by walking into one. Our own
/// leg of such a case is safe and compares normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReRecord {
    Allowed,
    Forbidden { reason: &'static str },
}

/// A case of the suite.
pub trait Case {
    /// The case's name: its recording's file name, and its row in the report.
    fn name(&self) -> &'static str;

    /// Where the case's Metal answer comes from.
    fn metal_source(&self) -> MetalSource;

    /// The shaders the case runs, as committed fixtures; empty where it runs none.
    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(Vec::new())
    }

    /// The fingerprint of everything the case feeds its run: the bytes it writes into each buffer
    /// and texture before it runs, its specialization constants, and the parameters it sets.
    ///
    /// The suite emits this itself, at the reserved site [`INPUTS_SITE`], and the comparison checks
    /// it before any output. A case whose inputs are not the recording's is
    /// [`Unrecorded`](crate::verdict::Verdict::Unrecorded), never a pass: its Metal answer was
    /// taken from other bytes. Without it a recording decays in silence the first time a fixture
    /// or a generator changes.
    fn inputs(&self) -> Result<Vec<Input>, String>;

    /// Set the case up on `side`, run it, and emit the sites it observes. A call that did not
    /// answer stops the run with an [`Abort`]; the sites emitted before it still compare.
    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort>;

    /// What the case loosens, and what justifies each. Empty is the default, and the default is
    /// what a case should keep: comparison is exact.
    fn allowances(&self) -> Vec<Allowance> {
        Vec::new()
    }

    /// The transcript the case's carried measurement states, and the runs it stands on, for a case
    /// whose [`MetalSource`] is [`MetalSource::Carried`]. `None` for a case record mode runs on
    /// Metal.
    ///
    /// A carried answer is the case stating what our own measurement of this chip found, in the
    /// shape the case emits, so it compares through the same path a recorded run does.
    fn carried_transcript(&self) -> Result<Option<(u32, Transcript)>, String> {
        Ok(None)
    }
}

/// The reserved site the suite emits a case's input fingerprint at.
pub const INPUTS_SITE: &str = "inputs";

/// One input a case feeds its run, and its digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// What the input is, as the case names it: `buffer:0`, `constant:width`, `texture:albedo`.
    pub name: String,
    pub digest: crate::transcript::Digest,
}

impl Input {
    /// The input `name`, fingerprinted by the SHA-256 of `bytes`.
    pub fn bytes(name: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            name: name.into(),
            digest: crate::transcript::Digest::sha256(bytes),
        }
    }
}

/// The node the suite emits for a case's inputs.
pub fn inputs_node(inputs: &[Input]) -> Node {
    Node::branch(
        INPUTS_SITE,
        inputs
            .iter()
            .map(|input| {
                Node::leaf(
                    input.name.clone(),
                    Observed::absent().digested(input.digest.clone()),
                )
            })
            .collect(),
    )
}

/// The site a call that answered an object emits, as a carried measurement states it: the call
/// answered, and nothing of what it answered. A carried transcript holds the same sites a run
/// emits, so it states this one too.
pub fn answered() -> Observed {
    Observed::value(Value::ok(Value::Absent))
}

/// Emit a site for `answer` and go on with what it answered, or stop the run.
///
/// `observed` says what the site records of a successful answer: `Value::Absent` for a call that
/// answers an object, whose identity is not observable.
pub fn call<T>(
    sites: &mut Sites,
    at: &str,
    answer: Answer<T>,
    observed: impl FnOnce(&T) -> Value,
) -> Result<T, Abort> {
    match answer {
        Ok(value) => {
            sites.value(at, Value::ok(observed(&value)));
            Ok(value)
        }
        Err(failure) => {
            sites.push(Node::leaf(at, Observed::value(failure.value.clone())));
            Err(Abort {
                at: at.to_owned(),
                failure,
            })
        }
    }
}

/// [`call`] for a call that answers an object: the site records that it answered, and nothing of
/// what it answered.
pub fn made<T>(sites: &mut Sites, at: &str, answer: Answer<T>) -> Result<T, Abort> {
    call(sites, at, answer, |_| Value::Absent)
}
