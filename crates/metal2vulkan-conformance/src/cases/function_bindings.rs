//! What a kernel's reflection says its arguments are: `MTLBinding` and the type tree under it.
//!
//! The suite compared one bit of a reflection before this family — `texture_accesses` reads a
//! render pipeline's texture slots and their access — and nothing else. The whole of
//! `MTLBufferBinding`, `MTLTextureBinding`, `MTLThreadgroupBinding`, `MTLStructType`,
//! `MTLStructMember`, `MTLPointerType`, `MTLArrayType` and `MTLTextureReferenceType` was compared
//! against Metal nowhere.
//!
//! For an argument buffer that tree **is** the layout contract: every member's `offset` and
//! `argumentIndex`, every array's `stride` and `argumentIndexStride`, every pointer's `alignment`
//! and `dataSize`. [`crate::cases::argument_encoders`] compares where an encoder puts each
//! argument; this compares what Metal says each argument is. Checking one without the other is
//! checking an answer without its question.
//!
//! # The binding kinds, measured
//!
//! `MTLBindingType` holds twelve values and the suite's eight hundred recordings held **four** of
//! them anywhere: `Buffer`, `ThreadgroupMemory`, `Texture` and `Sampler`. Of the other eight, four
//! are ray tracing's and one is Metal 4's tensor, all out of this project's scope; `ObjectPayload`
//! needs a mesh pipeline's reflection, which no case reads, and is a named gap. The two that are
//! reachable from a compute pipeline are here: a **visible-function table**, whose kernel takes one
//! as an argument, and an **imageblock**, whose kernel declares one explicitly.
//!
//! Five kernels, chosen so that between them the three subprotocols a binding can conform to are
//! all reached, and so that two binding types nothing had compared are:
//!
//! - **the argument-buffer kernel** (`px_samplers`' `px_sample_argument`): two plain device
//!   pointers, which answer a `bufferPointerType`, and one argument buffer, which answers a
//!   `bufferStructType` whose members are a texture reference and a sampler;
//! - **the texture-slot kernel** (`xr_samples`' `xr_sample_runtime`): a `[[texture(0)]]` and a
//!   `[[sampler(0)]]`, which is the only way to reach `MTLTextureBinding`'s own fields;
//! - **the threadgroup kernel** (`a27_ob1_threadgroup`): a `threadgroup uint *` argument, which is
//!   the only way to reach `MTLThreadgroupBinding`'s.
//!
//! # Why this family is safe
//!
//! Each case makes one compute pipeline state and reads its reflection. Nothing is dispatched,
//! nothing is allocated for the device, and no command buffer is made, so no case here can fault
//! the device or trip its watchdog. The pipeline creation is the one-time translation cost the
//! project's ceilings already cover.
//!
//! # Exactness
//!
//! Every field is a number, an enumeration or a name, and the family declares no tolerance. An
//! enumeration is recorded as its Metal enumerator name beside its ordinal, so a difference reads
//! `MTLDataTypeFloat2` against `MTLDataTypeFloat4` rather than 4 against 6.

use crate::case::{Abort, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::function_bindings::bindings_node;
use crate::recording::ShaderRef;
use crate::transcript::{Digest, Node, Observed, Sites};

/// The site the reflection's answer sits at.
const BINDINGS: &str = "bindings";

/// A case over one kernel's reflected arguments.
pub struct FunctionBindingsCase {
    name: &'static str,
    /// The fixture and the entry point whose reflection the case reads.
    fixture: &'static str,
    entry: &'static str,
}

/// Every case of the family.
pub fn all() -> Vec<FunctionBindingsCase> {
    vec![
        // The argument-buffer kernel: `constant PixelArguments &a [[buffer(2)]]` holding a texture
        // and a sampler, beside two plain device pointers at 0 and 1.
        FunctionBindingsCase {
            name: "function-bindings-argument-buffer",
            fixture: "px_samplers",
            entry: "px_sample_argument",
        },
        // The texture-slot kernel: `[[texture(0)]]` and `[[sampler(0)]]`, which is where
        // `MTLTextureBinding`'s own fields are.
        FunctionBindingsCase {
            name: "function-bindings-texture-slots",
            fixture: "xr_samples",
            entry: "xr_sample_runtime",
        },
        // The threadgroup-memory kernel: `threadgroup uint *tg [[threadgroup(0)]]`, which is where
        // `MTLThreadgroupBinding`'s are.
        FunctionBindingsCase {
            name: "function-bindings-threadgroup-memory",
            fixture: "a27_ob1_threadgroup",
            entry: "ob1_threadgroup",
        },
        // The visible-function-table kernel: `visible_function_table<WordOp> ops [[buffer(1)]]`
        // beside a plain `device uint *` at 0, which is the only shape in this tree whose
        // reflection can hold `MTLBindingTypeVisibleFunctionTable`. The kernel links no visible
        // function: it calls through the table's entries rather than naming one, so the pipeline
        // compiles from the library alone and the binding is the table's own argument.
        FunctionBindingsCase {
            name: "function-bindings-visible-function-table",
            fixture: "visible_table_caller",
            entry: "visible_table_dispatch",
        },
        // The explicit-imageblock kernel: `imageblock<Cell, imageblock_layout_explicit> img`
        // beside a `device uint4 *` at 0, which is the only shape in this tree that can reach
        // `MTLBindingTypeImageblock` or `MTLBindingTypeImageblockData`. Whether Metal lists a
        // compute kernel's imageblock as a binding at all is the measurement.
        FunctionBindingsCase {
            name: "function-bindings-imageblock",
            fixture: "a4_imageblock_extent",
            entry: "a4_imageblock_extent",
        },
    ]
}

impl FunctionBindingsCase {
    fn loaded(&self) -> Fixture {
        Fixture {
            name: self.fixture,
            entry: self.entry,
        }
    }
}

impl Case for FunctionBindingsCase {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.loaded();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", self.fixture),
                    entry: self.entry.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.loaded();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The entry point is the whole of the rest of what the case feeds: the reflection is a
            // fact about one function of the library.
            Input::bytes("entry", self.entry.as_bytes()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = self.loaded().loaded().map_err(|why| stop("library", why))?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        match side.compute_pipeline_bindings(library, self.entry) {
            Ok(bindings) => {
                sites.push(bindings_node(BINDINGS, &bindings));
                Ok(())
            }
            Err(failure) => {
                sites.push(Node::leaf(BINDINGS, Observed::value(failure.recorded())));
                Err(Abort {
                    at: BINDINGS.to_owned(),
                    failure,
                })
            }
        }
    }
}
