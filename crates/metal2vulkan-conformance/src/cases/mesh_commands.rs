//! Mesh draws in an indirect command buffer: what commands with pipeline states of their own draw.
//!
//! Ported from `runtime_lane/mesh_commands.rs`'s `cpu_own` row, whose expectation is a transcript
//! of a Metal probe's printed lines. That lane's own documentation says its table is an **M4**'s
//! (macOS 27.0.1), while the suite's oracle is this M3 Max, so running it through record mode is
//! also the first measurement of this probe on the chip its recording is keyed to.
//!
//! The row is an indirect command buffer of two commands the host encodes, over a pass of two
//! `RGBA32Uint` attachments, 8x8, cleared to zero and stored: command 0 sets `mesh_draws`' object
//! pipeline, binds the control words `(2, 1, 0, 0)` at object buffer 0 and draws 3 threadgroups of
//! 4 object threads and 2 mesh threads; command 1 sets its grid pipeline and draws 12 by 4 threads
//! in mesh threadgroups of 4 by 2. Both states are made with `supportIndirectCommandBuffers`, the
//! buffer inherits neither state nor buffers, and the control buffer is made resident for the
//! object, mesh and fragment stages before the execution.
//!
//! What the row observes is typed: each attachment's whole 8x8 as bytes under a layout that reads
//! them as 32-bit words, the drawn texel count, the calls of the pass, and the command buffer's
//! status and error.
//!
//! ## What of the lane is **not** ported, and why
//!
//! This lane is the largest of the nine probe-line files, and most of its rows cannot be cases
//! this round:
//!
//! - **Rows that end Metal's process**: the CPU's mesh draw on a buffer of draws of primitives,
//!   and a pipeline state set on a command of a buffer whose command types hold no draw of its
//!   kind. The lane's own documentation says so, and this round must run neither leg on Metal.
//!   They port as recorded-only rows with re-recording forbidden, which needs their measured
//!   tables carried; that is work this round did not do.
//! - **Rows where Metal's GPU faults**: a kernel's mesh draw on a buffer of draws of primitives,
//!   which the lane records as `status=5`. Same treatment, same reason.
//! - **Rows a kernel encodes** (`kernel_own`, `enc_own`, `kernel_<kind>_on_<kind>`, `enc_barrier`):
//!   these need a compute kernel writing the buffer through its GPU resource ID with an argument
//!   buffer, which is a vocabulary of its own that no case of batch 1 otherwise needs.
//! - **The racy row**: a command's draw without a barrier, where the lane accepts either order's
//!   words. That is a case whose recording holds the **set** of outcomes with frequencies, which
//!   record mode supports; it needs enough runs to state the frequencies honestly, which one
//!   recording pass cannot.
//!
//! What is ported is the one row whose whole path is public Metal, so its Metal leg is real.

use crate::allowance::Allowance;
use crate::case::{
    Abort, Case, CommandDraw, Failure, Input, MeshCommand, MeshFunctions, MetalSource, Side,
    TARGET_TEXEL,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};

/// The case's name.
pub const CPU_OWN_COMMANDS: &str = "mesh-commands-own-state";

/// The owned fixture the case runs.
const FIXTURE: &str = "mesh_draws";

/// The pass's side in texels.
const SIDE: u32 = 8;

/// The color attachments the pass holds.
const ATTACHMENTS: u32 = 2;

/// The fragment function both pipelines write their attachments through.
const FRAGMENT: &str = "mesh_draws_fragment";

/// The object function's control words, as the probe wrote them.
const CONTROL: [u32; 4] = [2, 1, 0, 0];

/// Command 0's draw: 3 threadgroups of 4 object threads and 2 mesh threads (the lane's
/// `OBJECT_DRAW`).
const OBJECT_DRAW: CommandDraw = CommandDraw::Threadgroups {
    groups: [3, 1, 1],
    object: [4, 1, 1],
    mesh: [2, 1, 1],
};

/// Command 1's draw: 12 by 4 threads in mesh threadgroups of 4 by 2 (the lane's `GRID_THREADS`).
const GRID_DRAW: CommandDraw = CommandDraw::Threads {
    threads: [12, 4, 1],
    object: [1, 1, 1],
    mesh: [4, 2, 1],
};

/// The row's site name, as the probe named the case.
const ROW: &str = "cpu_own";

/// A case that compares what an indirect command buffer's mesh draws drew with Metal's.
pub struct MeshCommands {
    name: &'static str,
}

/// The case.
pub fn own_state_commands() -> MeshCommands {
    MeshCommands {
        name: CPU_OWN_COMMANDS,
    }
}

impl MeshCommands {
    /// The fixture both sides read.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The object pipeline's functions, and the grid pipeline's.
    fn functions() -> [MeshFunctions<'static>; 2] {
        [
            MeshFunctions {
                object: Some("mesh_draws_object"),
                mesh: "mesh_draws_mesh",
                fragment: FRAGMENT,
                attachments: ATTACHMENTS,
            },
            MeshFunctions {
                object: None,
                mesh: "mesh_draws_grid",
                fragment: FRAGMENT,
                attachments: ATTACHMENTS,
            },
        ]
    }
}

/// The layout one attachment's texels are read under: the whole 8x8 of level 0 of slice 0, four
/// 32-bit words a texel, packed, so a diff names the texel's row and the word inside it.
fn attachment_layout() -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: 0,
            slice: 0,
            aspect: Aspect::Color,
            sample: 0,
            origin: [0, 0, 0],
            size: [SIDE, SIDE, 1],
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: SIDE * TARGET_TEXEL / 4,
        row_pitch: SIDE * TARGET_TEXEL,
        rows: SIDE,
        image_pitch: SIDE * SIDE * TARGET_TEXEL,
        images: 1,
        gaps: Gaps::Compared,
    }
}

/// The texels any attachment holds a non-zero word in: the probe's `drawn=`.
fn drawn(attachments: &[Vec<u8>]) -> u64 {
    let texel = TARGET_TEXEL as usize;
    let texels = (SIDE * SIDE) as usize;
    (0..texels)
        .filter(|at| {
            attachments.iter().any(|bytes| {
                bytes
                    .get(at * texel..(at + 1) * texel)
                    .is_some_and(|words| words.iter().any(|byte| *byte != 0))
            })
        })
        .count() as u64
}

impl Case for MeshCommands {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The pass's notices are a mesh pass's, plus the host step a device announces for commands
        // it draws itself; both are named by kind in `mesh_draws`.
        super::mesh_draws::mesh_notice_allowances()
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: "mesh_draws_mesh".to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let control: Vec<u8> = CONTROL.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut commands = Vec::new();
        for (functions, draw) in Self::functions().iter().zip([OBJECT_DRAW, GRID_DRAW]) {
            commands.extend(functions.object.unwrap_or("-").as_bytes());
            commands.push(b',');
            commands.extend(functions.mesh.as_bytes());
            commands.extend(format!(",{draw:?}\n").into_bytes());
        }
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:object:0", &control),
            Input::bytes(
                "render-targets",
                format!("{ATTACHMENTS}x RGBA32Uint {SIDE}x{SIDE} cleared to zero").as_bytes(),
            ),
            Input::bytes("commands", &commands),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let [object, grid] = Self::functions();
        let object = crate::case::made(
            sites,
            "object-pipeline",
            side.mesh_pipeline_for_commands(library, &object),
        )?;
        let grid = crate::case::made(
            sites,
            "grid-pipeline",
            side.mesh_pipeline_for_commands(library, &grid),
        )?;
        let words: Vec<u8> = CONTROL.iter().flat_map(|word| word.to_le_bytes()).collect();
        let control = crate::case::made(sites, "control", side.buffer(&words))?;
        let commands = crate::case::made(
            sites,
            "command-buffer",
            side.mesh_command_buffer(&[
                MeshCommand {
                    pipeline: object,
                    object_buffers: vec![(control, 0)],
                    mesh_buffers: Vec::new(),
                    draw: OBJECT_DRAW,
                },
                MeshCommand {
                    pipeline: grid,
                    object_buffers: Vec::new(),
                    mesh_buffers: Vec::new(),
                    draw: GRID_DRAW,
                },
            ]),
        )?;
        let mut targets = Vec::new();
        for at in 0..ATTACHMENTS {
            targets.push(
                side.render_target(SIDE, SIDE)
                    .map_err(|failure| stop(format!("{ROW}/render_target/{at}"), failure))?,
            );
        }

        let outcome = side
            .execute_mesh_commands(&targets, commands, &[control])
            .map_err(|failure| stop(format!("{ROW}/pass"), failure))?;
        let mut node = Node::branch(ROW, Vec::new());
        // The pass's calls in the order the side makes them: one `useResource` per resident
        // resource, the execution, and the commit.
        let calls: Vec<&str> = ["useResource", "executeCommandsInBuffer", "commit"].to_vec();
        let mut emitted = Node::branch("calls", Vec::new());
        for (index, refused) in outcome.refused.iter().enumerate() {
            let name = calls.get(index).copied().unwrap_or("call");
            emitted = emitted.with(Node::leaf(
                format!("{index}-{name}"),
                Observed::value(match refused {
                    Some(failure) => failure.value.clone(),
                    None => Value::ok(Value::Absent),
                }),
            ));
        }
        node = node.with(emitted);
        if let Some((index, failure)) = outcome
            .refused
            .iter()
            .enumerate()
            .find_map(|(index, refused)| refused.as_ref().map(|failure| (index, failure.clone())))
        {
            let name = calls.get(index).copied().unwrap_or("call");
            sites.push(node);
            return Err(Abort {
                at: format!("{ROW}/calls/{index}-{name}"),
                failure,
            });
        }
        node = node.with(Node::leaf(
            "status",
            Observed::value(Value::ok(outcome.status.clone())),
        ));
        node = node.with(Node::leaf(
            "command-buffer-error",
            Observed::value(match &outcome.error {
                Some(error) => Value::Variant {
                    name: "Some".to_owned(),
                    payload: Box::new(error.clone()),
                },
                None => Value::unit_variant("None"),
            }),
        ));

        let mut read = Vec::new();
        for (at, target) in targets.iter().enumerate() {
            read.push(
                side.read_texture(*target, [SIDE, SIDE], SIDE * TARGET_TEXEL)
                    .map_err(|failure| stop(format!("{ROW}/attachments/{at}"), failure))?,
            );
        }
        let mut attachments = Node::branch("attachments", Vec::new());
        for (at, bytes) in read.iter().enumerate() {
            attachments = attachments.with(Node::leaf(
                at.to_string(),
                Observed::bytes(bytes.clone(), attachment_layout()),
            ));
        }
        node = node.with(attachments);
        node = node.with(Node::leaf(
            "drawn",
            Observed::value(Value::Uint(drawn(&read))),
        ));
        sites.push(node);
        Ok(())
    }
}
