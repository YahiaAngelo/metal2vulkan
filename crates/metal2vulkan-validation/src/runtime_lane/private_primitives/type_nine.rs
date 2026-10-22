//! Device tests of a draw of Metal's private primitive type 9 ([`DrawPrimitiveType::Nine`]) on
//! the device the cfg selects, under the validation layer: its command buffer fails as Metal's
//! GPU fails it, and the queue's other command buffers answer as Metal's do.
//!
//! Metal's answers (probes on the Apple M3 Max, macOS 27.0.1, 21 runs, the scenario below and
//! one that also commits after the hang read `Scheduled` and 20 ms on):
//!
//! - **A direct draw of 3 vertices or more** (1 or 2 instances) hangs Metal's GPU: the command
//!   buffer reads `Scheduled` 5.06 to 13.2 ms after its commit and fails with
//!   `MTLCommandBufferErrorTimeout` ([`CommandBufferError::GpuHang`]) 32.7 to 42.7 ms after it.
//!   Its encoders before the pass run; the pass (its clear too) and every
//!   encoder after it run nothing; its event signal fires. Every command buffer of its queue
//!   scheduled meanwhile runs nothing and completes with no error together with it, its event
//!   signals firing: committed at once or 10 ms on, or with an event wait met during the hang by
//!   the host or by another queue's command buffer. One whose event wait the host met after the
//!   hang completed ran, as did one of another queue (in 4 ms, during the hang) and one committed
//!   after the hang.
//! - **A direct draw of 1 or 2 vertices, or of no vertex or instance** draws nothing: the pass
//!   clears, and every command buffer runs.
//! - **An indexed draw** (1, 2 or 3 indices) faults Metal's GPU: the command buffer fails with
//!   `MTLCommandBufferErrorPageFault` ([`CommandBufferError::PageFault`]); the pass runs nothing
//!   (no clear), its other encoders run and its signal fires, and the queue's others run.
//! - **An indirect draw** (of 3 vertices or none) **or an indexed indirect one** fails the command
//!   buffer as a victim of the GPU's recovery, `MTLCommandBufferErrorInternal`
//!   ([`CommandBufferError::GpuRecoveryVictim`]), with the indexed draw's other answers.
//!
//! Each scenario commits on the lane's queue the command buffer of the draw (a blit's mark, the
//! pass of the draw over a target the host filled, a mark, an event signal of 1, a mark), at once
//! one that marks and signals 2, and on another queue one that marks. 10 ms on, on the lane's
//! queue, it commits one that marks and three that wait for an event first, then mark: the host
//! meets the first's 15 ms after the draw's commit, another queue's command buffer committed then
//! the second's, and the host the third's only once the others completed. Last, it commits one
//! more that marks.

use super::super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use super::{Lane, FORMATS, SIDE};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{IndexType, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    DrawPrimitiveType, EventRef, LoadAction, Region, RenderPassDescriptor, RenderPipelineState,
    ResourceOptions, StoreAction, Texture, TextureDescriptor,
};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// The earliest Metal's command buffer its GPU hung at read `Scheduled` after its commit:
/// 5.06 ms.
const SCHEDULED: Duration = Duration::from_micros(5_060);

/// The earliest Metal failed a command buffer its GPU hung at after its commit: 32.7 ms.
const HANG: Duration = Duration::from_micros(32_700);

/// When the host meets the event wait of the queue's command buffer it meets during a hang,
/// after the draw's commit.
const DURING: Duration = Duration::from_millis(15);

/// The byte the host fills the pass's first target with, which its clear overwrites.
const FILLED: u8 = 0x5a;

/// The marks, by slot: each command buffer's, and the hanging one's three.
const MARKS: [(&str, u8); 10] = [
    ("before the pass", 0xa1),
    ("after the pass", 0xb1),
    ("after the signal", 0xb2),
    ("the queue's, at once", 0xc1),
    ("the queue's, 10 ms on", 0xc2),
    (
        "the queue's, its wait met by the host during the hang",
        0xc4,
    ),
    ("the queue's, its wait met by another queue's", 0xc5),
    ("the queue's, its wait met after the hang", 0xc3),
    ("another queue's", 0xd1),
    ("the one after", 0xe1),
];

/// The slots of [`MARKS`] that a scenario's command buffers other than the draw's mark, in the
/// order of [`Outcome::answers`] after the draw's.
const OTHERS: [usize; 7] = [3, 4, 5, 6, 7, 8, 9];

/// The draw of type 9 a scenario's pass makes.
#[derive(Clone, Copy, Debug)]
enum Draw {
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
    Direct { vertices: u32, instances: u32 },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:` of 16-bit
    /// indices 0, 1, 2.
    Indexed(u32),
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:` of this vertex count, one instance.
    Indirect(u32),
    /// `drawIndexedPrimitives:indexType:...:indirectBuffer:` of 3 16-bit indices.
    IndexedIndirect,
}

/// How a scenario's command buffer of the draw ends, as Metal's does.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expected {
    /// It draws nothing, and every command buffer runs.
    Nothing,
    /// Metal's GPU hangs at it.
    Hang,
    /// The pass runs nothing and the command buffer fails with the error.
    Fails(CommandBufferError),
}

/// One command buffer's answer.
#[derive(Debug)]
struct Answer {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// Its commit's instant.
    committed: Instant,
    /// When its scheduled handler ran.
    scheduled: Instant,
    /// When its completed handler ran.
    completed: Instant,
}

impl Answer {
    /// From its commit to its completed handler.
    fn took(&self) -> Duration {
        self.completed - self.committed
    }

    /// From its commit to its scheduled handler.
    fn to_scheduled(&self) -> Duration {
        self.scheduled - self.committed
    }

    /// It completed with no error.
    fn completed_cleanly(&self) -> bool {
        self.status == CommandBufferStatus::Completed && self.error.is_none()
    }
}

/// A committed command buffer and the instants its scheduled and completed handlers ran.
struct Tracked {
    command_buffer: CommandBuffer,
    committed: Option<Instant>,
    scheduled: Arc<OnceLock<Instant>>,
    completed: Arc<OnceLock<Instant>>,
}

impl Tracked {
    fn new(queue: &CommandQueue) -> Tracked {
        let command_buffer = queue.command_buffer().unwrap();
        let (scheduled, completed) = (Arc::new(OnceLock::new()), Arc::new(OnceLock::new()));
        let set = Arc::clone(&scheduled);
        command_buffer
            .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                let _ = set.set(Instant::now());
            }))
            .unwrap();
        let set = Arc::clone(&completed);
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                let _ = set.set(Instant::now());
            }))
            .unwrap();
        Tracked {
            command_buffer,
            committed: None,
            scheduled,
            completed,
        }
    }

    /// A blit that fills `marks`' slot `slot` with its mark.
    fn mark(&self, marks: &Buffer, slot: usize) {
        let mut blit = self.command_buffer.blit_command_encoder().unwrap();
        let at = slot as u64 * 4;
        blit.fill_buffer(marks, at..at + 4, MARKS[slot].1).unwrap();
        blit.end_encoding().unwrap();
    }

    fn commit(&mut self) {
        self.committed = Some(Instant::now());
        self.command_buffer.commit().unwrap();
    }

    fn wait(&self) {
        self.command_buffer.wait_until_completed_for(WAIT).unwrap();
    }

    fn answer(&self) -> Answer {
        Answer {
            status: self.command_buffer.status(),
            error: self.command_buffer.error(),
            committed: self.committed.unwrap(),
            scheduled: *self.scheduled.get().expect("its scheduled handler ran"),
            completed: *self.completed.get().expect("its completed handler ran"),
        }
    }
}

/// A scenario's answers: the command buffer of the draw's, then the others' in [`OTHERS`]'s order
/// (the queue's at once, 10 ms on, its wait met during the hang by the host and by another
/// queue's, its wait met after the hang; another queue's, the one after), the marks, the event's
/// value and whether the pass cleared its target.
struct Outcome {
    answers: [Answer; 8],
    marks: [u8; 10],
    event: u64,
    cleared: bool,
}

/// The pass's targets, the first filled with [`FILLED`], and the pass that clears them.
fn targets(lane: &Lane) -> (Vec<Texture>, RenderPassDescriptor) {
    let targets: Vec<_> = FORMATS
        .iter()
        .map(|format| {
            let mut descriptor = TextureDescriptor::texture_2d(*format, SIDE, SIDE, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = ResourceOptions::shared();
            lane.device.new_texture(&descriptor).unwrap()
        })
        .collect();
    let filled = vec![FILLED; (SIDE * SIDE * 16) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    targets[0]
        .replace_region(region, 0, 0, &filled, SIDE * 16, SIDE * SIDE * 16)
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    for (i, target) in targets.iter().enumerate() {
        let color = &mut pass.color_attachments[i];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::Store;
    }
    (targets, pass)
}

/// Run the scenario (the module docs) of `draw` under `pipeline`.
fn scenario(lane: &Lane, pipeline: &RenderPipelineState, draw: Draw) -> Outcome {
    let device = &lane.device;
    let marks = buffer_with(device, &[0; 40]).unwrap();
    let event = device.new_shared_event().unwrap();
    let during_gate = device.new_shared_event().unwrap();
    let gpu_gate = device.new_shared_event().unwrap();
    let gate = device.new_shared_event().unwrap();
    let other_queue = device.new_command_queue().unwrap();
    let indices = buffer_with(device, &[0, 0, 1, 0, 2, 0, 0, 0]).unwrap();
    let words: &[u32] = match draw {
        Draw::Indirect(vertices) => &[vertices, 1, 0, 0],
        _ => &[3, 1, 0, 0, 0],
    };
    let arguments: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let arguments = buffer_with(device, &arguments).unwrap();
    let (targets, pass) = targets(lane);

    let mut drawn = Tracked::new(&lane.queue);
    drawn.mark(&marks, 0);
    let mut encoder = drawn.command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder
        .set_vertex_buffer(Some(&lane.positions), 0, 0)
        .unwrap();
    encoder
        .set_vertex_buffer(Some(&lane.offsets), 0, 1)
        .unwrap();
    let nine = DrawPrimitiveType::Nine;
    match draw {
        Draw::Direct {
            vertices,
            instances,
        } => encoder.draw_primitives(nine, 0, vertices, instances, 0),
        Draw::Indexed(count) => {
            encoder.draw_indexed_primitives(nine, count, IndexType::UInt16, &indices, 0, 1, 0, 0)
        }
        Draw::Indirect(_) => encoder.draw_primitives_indirect(nine, &arguments, 0),
        Draw::IndexedIndirect => encoder.draw_indexed_primitives_indirect(
            nine,
            IndexType::UInt16,
            &indices,
            0,
            &arguments,
            0,
        ),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    drawn.mark(&marks, 1);
    drawn
        .command_buffer
        .encode_signal_event(EventRef::from(&event), 1)
        .unwrap();
    drawn.mark(&marks, 2);

    let mut at_once = Tracked::new(&lane.queue);
    at_once.mark(&marks, 3);
    at_once
        .command_buffer
        .encode_signal_event(EventRef::from(&event), 2)
        .unwrap();
    let mut later = Tracked::new(&lane.queue);
    later.mark(&marks, 4);
    let waiting = |slot: usize, gate| {
        let tracked = Tracked::new(&lane.queue);
        tracked
            .command_buffer
            .encode_wait_for_event(EventRef::from(gate), 1)
            .unwrap();
        tracked.mark(&marks, slot);
        tracked
    };
    let mut during = waiting(5, &during_gate);
    let mut on_gpu = waiting(6, &gpu_gate);
    let mut held = waiting(7, &gate);
    let mut other = Tracked::new(&other_queue);
    other.mark(&marks, 8);
    let signaller = other_queue.command_buffer().unwrap();
    signaller
        .encode_signal_event(EventRef::from(&gpu_gate), 1)
        .unwrap();

    drawn.commit();
    at_once.commit();
    other.commit();
    std::thread::sleep(Duration::from_millis(10));
    later.commit();
    // The event waits last in the queue's order, so they hold back none of the others.
    during.commit();
    on_gpu.commit();
    held.commit();
    signaller.commit().unwrap();
    let due = drawn.committed.unwrap() + DURING;
    if let Some(left) = due.checked_duration_since(Instant::now()) {
        std::thread::sleep(left);
    }
    during_gate.set_signaled_value(1);
    for tracked in [&drawn, &at_once, &later, &during, &on_gpu] {
        tracked.wait();
    }
    gate.set_signaled_value(1);
    held.wait();
    other.wait();
    signaller.wait_until_completed_for(WAIT).unwrap();
    let mut after = Tracked::new(&lane.queue);
    after.mark(&marks, 9);
    after.commit();
    after.wait();

    let read = read_buffer(&marks, 0, 40).unwrap();
    let mut texel = [0u8; 16];
    targets[0]
        .get_bytes(&mut texel, 16, 16, Region::new_2d(0, 0, 1, 1), 0, 0)
        .unwrap();
    Outcome {
        answers: [
            &drawn, &at_once, &later, &during, &on_gpu, &held, &other, &after,
        ]
        .map(Tracked::answer),
        marks: std::array::from_fn(|slot| read[slot * 4]),
        event: event.signaled_value(),
        cleared: texel == [0; 16],
    }
}

/// Run each of `draws`'s scenarios on the test device and check it against `expected`.
fn answers_as_metal_does(draws: &[Draw], expected: Expected) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let pipeline = lane.pipeline("pt_plain");
        for &draw in draws {
            let outcome = scenario(&lane, &pipeline, draw);
            check(&outcome, expected, &format!("{draw:?} on {name}"));
            let [drawn, rest @ ..] = &outcome.answers;
            let ms = |answer: &Answer| answer.took().as_secs_f64() * 1e3;
            let rest: Vec<String> = rest
                .iter()
                .zip(OTHERS)
                .map(|(answer, slot)| {
                    let apart = match answer.completed.checked_duration_since(drawn.completed) {
                        Some(after) => after.as_secs_f64() * 1e3,
                        None => -(drawn.completed - answer.completed).as_secs_f64() * 1e3,
                    };
                    format!(
                        "{} {:.1} ms ({apart:+.2} ms from the draw's, Scheduled at {:.1} ms)",
                        MARKS[slot].0,
                        ms(answer),
                        answer.to_scheduled().as_secs_f64() * 1e3,
                    )
                })
                .collect();
            println!(
                "{name}: {draw:?}: {:?} {:?} in {:.1} ms, Scheduled at {:.1} ms; {}",
                drawn.status,
                drawn.error,
                ms(drawn),
                drawn.to_scheduled().as_secs_f64() * 1e3,
                rest.join("; "),
            );
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Check `outcome` against Metal's answer `expected` (the module docs).
fn check(outcome: &Outcome, expected: Expected, label: &str) {
    let [drawn, at_once, later, during, on_gpu, held, other, after] = &outcome.answers;
    let marked = |slots: &[usize]| -> [u8; 10] {
        std::array::from_fn(|slot| match slots.contains(&slot) {
            true => MARKS[slot].1,
            false => 0,
        })
    };
    assert_eq!(outcome.event, 2, "{label}: the event's value");
    let discarded = [
        (at_once, "at once"),
        (later, "10 ms on"),
        (during, "met by the host during the hang"),
        (on_gpu, "met by another queue's"),
    ];
    for (answer, which) in
        discarded
            .into_iter()
            .chain([(held, "held"), (other, "another queue's"), (after, "after")])
    {
        assert!(answer.completed_cleanly(), "{label}: {which}: {answer:?}");
    }
    match expected {
        Expected::Nothing => {
            assert!(drawn.completed_cleanly(), "{label}: {drawn:?}");
            assert!(outcome.cleared, "{label}: the pass cleared its target");
            assert_eq!(
                outcome.marks,
                marked(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "{label}"
            );
        }
        Expected::Hang => {
            assert_eq!(drawn.status, CommandBufferStatus::Error, "{label}");
            assert_eq!(drawn.error, Some(CommandBufferError::GpuHang), "{label}");
            assert!(
                drawn.took() >= HANG,
                "{label}: failed in {:?}",
                drawn.took()
            );
            let scheduled = drawn.to_scheduled();
            assert!(
                (SCHEDULED..HANG).contains(&scheduled),
                "{label}: read Scheduled {scheduled:?} after its commit"
            );
            assert!(!outcome.cleared, "{label}: the pass ran nothing");
            assert_eq!(outcome.marks, marked(&[0, 7, 8, 9]), "{label}");
            for (answer, which) in discarded {
                assert!(
                    answer.completed >= drawn.completed,
                    "{label}: {which} completed before the hang did"
                );
            }
            assert!(
                other.completed < drawn.completed,
                "{label}: another queue's command buffer waited for the hang"
            );
        }
        Expected::Fails(error) => {
            assert_eq!(drawn.status, CommandBufferStatus::Error, "{label}");
            assert_eq!(drawn.error, Some(error), "{label}");
            assert!(!outcome.cleared, "{label}: the pass ran nothing");
            assert_eq!(
                outcome.marks,
                marked(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "{label}"
            );
        }
    }
}

/// A direct draw of type 9 of 3 vertices or more hangs Metal's GPU: its command buffer reads
/// `Scheduled` no sooner than Metal's did, long before it fails with `GpuHang` no sooner than
/// Metal's did, running nothing from the pass on but its event
/// signal, and the queue's command buffers committed meanwhile run nothing and complete cleanly
/// after it, those whose event waits were met during the hang too, while one whose wait is met
/// after it, another queue's and a later one run.
#[test]
fn a_direct_draw_of_type_nine_hangs_as_metal_s_gpu_does() {
    answers_as_metal_does(
        &[
            Draw::Direct {
                vertices: 3,
                instances: 1,
            },
            Draw::Direct {
                vertices: 4,
                instances: 1,
            },
            Draw::Direct {
                vertices: 3,
                instances: 2,
            },
        ],
        Expected::Hang,
    );
}

/// A direct draw of type 9 of fewer than 3 vertices, or of no instance, draws nothing: the pass
/// clears and every command buffer runs.
#[test]
fn a_direct_draw_of_type_nine_of_fewer_vertices_draws_nothing() {
    let direct = |vertices, instances| Draw::Direct {
        vertices,
        instances,
    };
    answers_as_metal_does(
        &[direct(0, 1), direct(1, 1), direct(2, 1), direct(3, 0)],
        Expected::Nothing,
    );
}

/// An indexed draw of type 9 fails its command buffer with `PageFault`, and an indirect one with
/// `GpuRecoveryVictim`: the pass runs nothing, everything else runs.
#[test]
fn indexed_draws_of_type_nine_fault_and_indirect_ones_are_victims() {
    answers_as_metal_does(
        &[Draw::Indexed(1), Draw::Indexed(2), Draw::Indexed(3)],
        Expected::Fails(CommandBufferError::PageFault),
    );
    answers_as_metal_does(
        &[Draw::Indirect(3), Draw::Indirect(0), Draw::IndexedIndirect],
        Expected::Fails(CommandBufferError::GpuRecoveryVictim),
    );
}
