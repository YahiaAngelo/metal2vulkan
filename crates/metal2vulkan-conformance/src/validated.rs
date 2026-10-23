//! The second Metal oracle: the same case under Metal's own API validation layer.
//!
//! Plain Metal is the behavioural oracle, because that is how a consumer runs the runtime. What
//! Metal's API validation does with the same case is a **second recorded fact**, never compared
//! with our behaviour.
//!
//! # A validated stop is not Metal refusing the call
//!
//! The ruling, and the reason the rest of this module exists: Metal's API validation stopping a
//! call is **not** Metal refusing it. Plain Metal is the oracle, and the owner's standing rule is
//! that what Metal allows we must allow. So where plain Metal answers a call and we return a
//! typed error, that is a gap of ours, whatever its validation layer thinks: the row is
//! [`Verdict::Refused`](crate::verdict::Verdict::Refused), it counts in the suite's own number,
//! and it is never compliant. The suite calling such a row a pass would be the suite laundering a
//! refusal.
//!
//! What the validated fact is for is the record and the report. It may **document** a refusal
//! ([`crate::allowance::Justification::MetalValidation`]) — a refusal Metal's own validation also
//! refuses reads differently from one Metal is happy with, and the first is the better bug report
//! — and it is still held to its evidence, which is what keeps this leg load-bearing rather than
//! decorative. It buys an explanation, never a pass.
//!
//! # Why a child process, and why `nslog` rather than the default
//!
//! Metal reads `MTL_DEBUG_LAYER` as it initialises, so the layer cannot be turned on inside a
//! process that has already made a device: the plain leg and the validated leg cannot share one.
//! More importantly, **Metal's validation reports by assertion**, and its default error mode ends
//! the process. Measured on this Mac (an Apple M3 Max, macOS 27.0.1): a rate-map descriptor of no
//! layers under `MTL_DEBUG_LAYER=1` alone printed
//! `` failed assertion `RasterizationRateMap Layers Validation ...` `` and died with SIGABRT,
//! while the same descriptor under [`ERROR_MODE`] printed the same words through `NSLog` and let
//! the call run. The valid modes are `ignore`, `assert`, `nslog`, `oslog` and `exception`;
//! `nslog` is the one that reports every check without ending anything.
//!
//! So the leg runs in a child process of this same command, under [`ERROR_MODE`] and
//! [`WARNING_MODE`], and the parent reads what the child's Metal said.
//!
//! **This is settled; do not re-litigate it from the letter of the safety rule.** That rule reads
//! that a shape whose Metal leg aborts is a recorded-only row that must never be re-run, and read
//! literally it would forbid re-recording even the *plain* leg of these cases, where Metal answers
//! cleanly and safely. Its purpose is that no case of ours may end Metal's process or fault the
//! device on demand, and this arrangement serves that purpose while keeping both legs
//! re-recordable:
//!
//! - **The suite's own process never has the validation layer on**, so no assertion of Apple's can
//!   end a record run, whatever a future case asks Metal. A child that aborts cannot take the
//!   recorder with it.
//! - **The layer reports instead of asserting**, so in practice nothing aborts at all.
//! - **Nothing of the validated leg reaches the GPU** for the cases that use it: a descriptor is
//!   compiled and queried on the CPU, and no command buffer is committed. A validation complaint
//!   is a CPU-side check, so the leg cannot fault or hang the device.
//!
//! If a child dies anyway, that is reported rather than guessed at: a leg that produced no
//! complaint and no clean exit is [`LegError::Died`], and record mode refuses the case instead of
//! writing a [`Validated`] value nobody measured. There is deliberately no `Validated` variant for
//! "taken and died": refusing the case is the honest answer, and a variant nothing constructs
//! would be dead code.
//!
//! # How a complaint is attributed to a call
//!
//! The child announces each [`Side`] call it is about to make on its standard
//! error ([`MARKER`]), so Metal's own output and our markers interleave in one stream in the order
//! they happened. Everything between two markers is the complaint of the call the first marker
//! names. Nothing is matched against Apple's wording, so a new check in a release the suite has
//! not seen is still attributed and still recorded.
//!
//! The message a recording keeps is stripped of `NSLog`'s timestamp and process prefix, because a
//! recording holds nothing host-specific: the prefix carries this run's process id.

use crate::case::{
    Answer, Bindings, Fill, Handle, IndirectReadBack, MeshCommand, MeshFunctions, MeshStep,
    PassOutcome, PlaneTexture, RateMapFacts, RateMapRequest, SamplerSpec, ShaderSource, Side,
    StageInRegion, StageInput, TextureSpec, TileFacts, TilePipelineSpec, TileStep, Venue,
};
use crate::recording::Validated;
use crate::transcript::Value;
use metal2vulkan::metal::PixelFormat;

/// `MTL_DEBUG_LAYER`: Metal's API validation layer.
pub const DEBUG_LAYER: (&str, &str) = ("MTL_DEBUG_LAYER", "1");

/// `MTL_DEBUG_LAYER_ERROR_MODE`: how the layer reports an error. `nslog` is the one mode that
/// reports every check without ending the process; the default, `assert`, aborts.
pub const ERROR_MODE: (&str, &str) = ("MTL_DEBUG_LAYER_ERROR_MODE", "nslog");

/// `MTL_DEBUG_LAYER_WARNING_MODE`: the same for a warning, so a warning is recorded too rather
/// than dropped.
pub const WARNING_MODE: (&str, &str) = ("MTL_DEBUG_LAYER_WARNING_MODE", "nslog");

/// The verb the child runs: it takes one case's validated leg and says nothing on its standard
/// output. It is not part of the command's documented surface; the parent spawns it.
pub const LEG_VERB: &str = "validated-leg";

/// What the child prints before each call it makes, so the parent can attribute a complaint to the
/// call it belongs to.
pub const MARKER: &str = "[conformance] call ";

/// The line Metal prints as the layer comes up, which is not a complaint about anything.
const BANNER: &str = "Metal API Validation";

/// Why the validated leg produced no answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegError {
    /// The child could not be started.
    NotStarted(String),
    /// The child ended other than cleanly and said nothing a complaint could be read out of, so
    /// what Metal's validation did is unknown rather than `Ran`.
    Died { status: String, output: String },
}

impl std::fmt::Display for LegError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegError::NotStarted(why) => {
                write!(out, "the validated leg could not be started: {why}")
            }
            LegError::Died { status, output } => write!(
                out,
                "the validated leg ended {status} with no validation message, so what Metal's \
                 API validation did is unknown:\n{output}"
            ),
        }
    }
}

/// What Metal's API validation did with `case`, taken by running the case in a child of this
/// command with the layer on.
#[cfg(target_os = "macos")]
pub fn take(case: &dyn crate::case::Case) -> Result<Validated, LegError> {
    let program = std::env::current_exe()
        .map_err(|error| LegError::NotStarted(format!("this command's own path: {error}")))?;
    let output = std::process::Command::new(&program)
        .arg(LEG_VERB)
        .arg(case.name())
        .env(DEBUG_LAYER.0, DEBUG_LAYER.1)
        .env(ERROR_MODE.0, ERROR_MODE.1)
        .env(WARNING_MODE.0, WARNING_MODE.1)
        .output()
        .map_err(|error| LegError::NotStarted(format!("{}: {error}", program.display())))?;
    let said = String::from_utf8_lossy(&output.stderr).into_owned();
    match read(&said) {
        Some(validated) => Ok(validated),
        None if output.status.success() => Ok(Validated::Ran),
        None => Err(LegError::Died {
            status: format!("{}", output.status),
            output: said,
        }),
    }
}

/// The first complaint `said` holds, as the recording keeps it: the call the marker before it
/// named, and the words Metal used with the host's own prefix stripped.
///
/// `None` where the layer complained about nothing, which is [`Validated::Ran`] for a child that
/// also finished.
pub fn read(said: &str) -> Option<Validated> {
    let mut call = String::new();
    let mut complaint: Vec<String> = Vec::new();
    for line in said.lines() {
        if let Some(named) = line.strip_prefix(MARKER) {
            if !complaint.is_empty() {
                break;
            }
            call = named.trim().to_owned();
            continue;
        }
        let text = strip_prefix(line);
        if text.is_empty() || text.starts_with(BANNER) {
            continue;
        }
        complaint.push(text.to_owned());
    }
    (!complaint.is_empty()).then(|| Validated::Stopped {
        call: if call.is_empty() {
            "before the first call".to_owned()
        } else {
            call
        },
        message: complaint.join("\n"),
    })
}

/// `line` without `NSLog`'s `<date> <time> <program>[<pid>:<tid>] ` prefix, which names this host's
/// process and must not reach a recording. A line without one is its own text.
fn strip_prefix(line: &str) -> &str {
    let Some((head, rest)) = line.split_once("] ") else {
        return line.trim_end();
    };
    if head.contains('[')
        && head
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_digit())
    {
        return rest.trim_end();
    }
    line.trim_end()
}

/// A side that announces each call it is about to make, so the parent can attribute Metal's own
/// output to the call that provoked it. It observes nothing and changes nothing else.
pub struct Announcing<S> {
    inner: S,
}

impl<S: Side> Announcing<S> {
    /// `inner`, announcing its calls.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Announce `call` and go on.
    fn at(&self, call: &str) {
        eprintln!("{MARKER}{call}");
    }
}

/// Every call is forwarded, and every call is announced.
///
/// **A forwarded method that is left out is a defect, not a gap**: `Side` gives most of its calls a
/// default that answers [`Failure::unavailable`](crate::case::Failure::unavailable), so an
/// unforwarded method would make `Announcing<MetalSide>` answer "no such call" where `MetalSide`
/// answers perfectly well, and the compiler would say nothing. `the_announcing_side_forwards_every_call`
/// catches that by running the whole case set through both and comparing the transcripts, so the
/// trait can keep growing without this wrapper rotting.
impl<S: Side> Side for Announcing<S> {
    fn venue(&self) -> Venue {
        self.inner.venue()
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        self.at("library");
        self.inner.library(source)
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        self.at("compute_pipeline");
        self.inner.compute_pipeline(library, entry)
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        self.at("compute_pipeline_with_stage_input");
        self.inner
            .compute_pipeline_with_stage_input(library, entry, stage_input)
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        self.at("buffer");
        self.inner.buffer(bytes)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        self.at("zeroed_buffer");
        self.inner.zeroed_buffer(length)
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        self.at("texture_2d");
        self.inner.texture_2d(format, size, row_pitch, bytes)
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        self.at("read_texture");
        self.inner.read_texture(texture, size, row_pitch)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        self.at("texture_2d_in_role");
        self.inner.texture_2d_in_role(spec, bytes)
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        self.at("replace_region");
        self.inner.replace_region(texture, size, row_pitch, bytes)
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        self.at("sampler");
        self.inner.sampler(spec)
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.at("dispatch");
        self.inner
            .dispatch(pipeline, bindings, threadgroups, threads)
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.at("dispatch_threads");
        self.inner
            .dispatch_threads(pipeline, bindings, stage_in, grid, threads)
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        self.at("wait");
        self.inner.wait(command_buffer)
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        self.at("read_buffer");
        self.inner.read_buffer(buffer, offset, length)
    }

    fn announced(&mut self) -> Vec<String> {
        self.inner.announced()
    }

    fn indirect_draw_read_back(
        &mut self,
        primitive_type: u64,
        indexed: bool,
    ) -> Answer<IndirectReadBack> {
        self.at("indirect_draw_read_back");
        self.inner.indirect_draw_read_back(primitive_type, indexed)
    }

    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        self.at("work");
        self.inner.work(work)
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        self.at("mesh_pipeline");
        self.inner.mesh_pipeline(library, functions)
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        self.at("mesh_pipeline_for_commands");
        self.inner.mesh_pipeline_for_commands(library, functions)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        self.at("render_target");
        self.inner.render_target(width, height)
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        self.at("mesh_pass");
        self.inner.mesh_pass(targets, steps)
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        self.at("mesh_command_buffer");
        self.inner.mesh_command_buffer(commands)
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        self.at("execute_mesh_commands");
        self.inner.execute_mesh_commands(targets, commands, used)
    }

    fn plane_texture(&mut self, buffer: Handle, over: &PlaneTexture) -> Answer<Handle> {
        self.at("plane_texture");
        self.inner.plane_texture(buffer, over)
    }

    fn fill_texture(&mut self, texture: Handle, fill: &Fill) -> Answer<Value> {
        self.at("fill_texture");
        self.inner.fill_texture(texture, fill)
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        self.at("rate_map");
        self.inner.rate_map(request)
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        self.at("rate_map_facts");
        self.inner.rate_map_facts(map, layers)
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        self.at("copy_rate_map_parameters");
        self.inner.copy_rate_map_parameters(map, into, offset)
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        self.at("tile_pipeline");
        self.inner.tile_pipeline(library, spec)
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        self.at("tile_facts");
        self.inner.tile_facts(state)
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        self.at("tile_pass");
        self.inner.tile_pass(targets, tile, steps)
    }
}
