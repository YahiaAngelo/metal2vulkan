//! The first seed case: the bytes a compute pass wrote.
//!
//! One owned fixture, one compute pipeline, one output buffer, one dispatch, one read-back. Every
//! word of the output compares against Metal's, and the diff names the thread's row and the word
//! inside it rather than a byte offset.
//!
//! The seed is `a1_srgb_decode_all_bytes`: Metal's sRGB decode of every one of the 256 byte values,
//! in `float` and in `half`, six words per thread. It is exhaustive over its whole input space,
//! which is where a rounding difference hides and where one word moving is a readable failure. It
//! reads no texture and no input buffer, binds one buffer, and cannot fault a GPU.

use crate::case::{call, made, Abort, Case, Failure, Input, MetalSource, ShaderSource, Side};
use crate::recording::ShaderRef;
use crate::replay::NOTICES_SITE;
use crate::transcript::{
    ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Transcript, Value,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

/// The case's name: its recording's file name, and its row in the report.
pub const SRGB_DECODE: &str = "pass-bytes-srgb-decode-all-bytes";

/// The threads the seed runs: one per byte value.
const THREADS: u32 = 256;

/// The words each thread writes: four `float` lanes, then two words of four `half` lanes.
const WORDS_PER_THREAD: u32 = 6;

/// The threads in one threadgroup.
const THREADS_PER_GROUP: u32 = 64;

/// A case that compares the bytes a compute pass wrote with Metal's.
pub struct PassBytes {
    name: &'static str,
    /// The owned fixture, named without an extension: its `.ll` is what `llvm-as` assembles for
    /// the runtime, and its `.air.ll` what Metal's assembler reads.
    fixture: &'static str,
    entry: &'static str,
    /// Threads in the grid, and words each writes.
    threads: u32,
    words_per_thread: u32,
}

/// The seed case.
pub fn srgb_decode_all_bytes() -> PassBytes {
    PassBytes {
        name: SRGB_DECODE,
        fixture: "a1_srgb_decode_all_bytes",
        entry: "a1_srgb_decode_all_bytes",
        threads: THREADS,
        words_per_thread: WORDS_PER_THREAD,
    }
}

impl PassBytes {
    /// The fixture's committed text in the dialect `extension` names.
    fn source(&self, extension: &str) -> Result<String, String> {
        let path = engine_dir().join(format!("{}{extension}", self.fixture));
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))
    }

    /// The fixture's bitcode, as `llvm-as` assembles its `.ll`.
    fn bitcode(&self) -> Result<Vec<u8>, String> {
        let path = engine_dir().join(format!("{}.ll", self.fixture));
        llvm_assemble(&path).map_err(|error| match error {
            AssembleError::Missing(why) => format!("no llvm-as: {why}"),
            AssembleError::Rejected(why) => format!("llvm-as rejected {}: {why}", self.fixture),
        })
    }

    /// The bytes the pass writes.
    fn output_bytes(&self) -> u64 {
        u64::from(self.threads) * u64::from(self.words_per_thread) * 4
    }

    /// The threadgroups and the threads in each.
    fn dispatch(&self) -> ([u32; 3], [u32; 3]) {
        (
            [self.threads / THREADS_PER_GROUP, 1, 1],
            [THREADS_PER_GROUP, 1, 1],
        )
    }

    /// The layout the output's words are read under: one row per thread, so a diff names the byte
    /// value whose decode moved and the word inside its row. The rows are packed, so there are no
    /// gaps to compare.
    fn layout(&self) -> ByteLayout {
        let pitch = self.words_per_thread * 4;
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: self.output_bytes(),
            },
            element: Element::U32,
            row_elements: self.words_per_thread,
            row_pitch: pitch,
            rows: self.threads,
            image_pitch: pitch * self.threads,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }

    /// The sites the case emits, with `output` the bytes the pass wrote and `status` the command
    /// buffer's status: the shape both sides produce, so a carried or hand-built answer is built
    /// through the same function the run uses.
    pub fn transcript(&self, output: Vec<u8>, status: &str) -> Result<Transcript, String> {
        let mut sites = Sites::new();
        sites.push(crate::case::inputs_node(&self.inputs()?));
        sites.value("library", Value::ok(Value::Absent));
        sites.value("pipeline", Value::ok(Value::Absent));
        sites.value("buffer/0", Value::ok(Value::Absent));
        sites.value("dispatch", Value::ok(Value::Absent));
        sites.value("status", Value::ok(Value::unit_variant(status)));
        sites.value("read-back", Value::ok(Value::Absent));
        sites.push(Node::leaf("output", Observed::bytes(output, self.layout())));
        sites.notices(NOTICES_SITE, &[] as &[&str]);
        Ok(sites.finish())
    }
}

impl Case for PassBytes {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        Ok(vec![
            ShaderRef {
                fixture: format!("{}.ll", self.fixture),
                entry: self.entry.to_owned(),
                source_sha256: crate::sha256_hex(self.source(".ll")?.as_bytes()),
            },
            ShaderRef {
                fixture: format!("{}.air.ll", self.fixture),
                entry: self.entry.to_owned(),
                source_sha256: crate::sha256_hex(self.source(".air.ll")?.as_bytes()),
            },
        ])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let (groups, threads) = self.dispatch();
        let mut words = Vec::new();
        for axis in groups.iter().chain(&threads) {
            words.extend(axis.to_le_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(self.source(".ll")?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(self.source(".air.ll")?.as_bytes()),
            },
            Input::bytes("dispatch", &words),
            // The output buffer starts zeroed on both sides, so the bytes the pass did not write
            // are part of the answer rather than whatever was there.
            Input::bytes("buffer:0", &vec![0u8; self.output_bytes() as usize]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let bitcode = self.bitcode().map_err(|why| stop("library", why))?;
        let air_text = self.source(".air.ll").map_err(|why| stop("library", why))?;
        let source = ShaderSource {
            bitcode: &bitcode,
            air_text: &air_text,
            entry: self.entry,
        };
        let (groups, threads) = self.dispatch();
        let length = self.output_bytes() as usize;

        let library = made(sites, "library", side.library(&source))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, self.entry),
        )?;
        let output = made(sites, "buffer/0", side.zeroed_buffer(length))?;
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch(pipeline, &[output], groups, threads),
        )?;
        call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        let bytes = call(
            sites,
            "read-back",
            side.read_buffer(output, 0, length),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("output", Observed::bytes(bytes, self.layout())));
        Ok(())
    }
}
