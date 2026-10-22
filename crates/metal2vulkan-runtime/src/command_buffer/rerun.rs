//! The check after a predicated loop laid down as copies (`predication.rs`): where the step past
//! its last copy halted its unit (the loop would run another pass), the check runs the loop's
//! laying again, from the halted loop's copy on (the halt word, `bind::PredicationOp::WhileFirst`),
//! until the loop ends, as Metal runs a loop on; then the command buffer goes on. Metal never ends
//! a loop that runs without bound (an M3 Max with macOS 27.0.1 ran an endless While and DoWhile for
//! 180 s, each, with no error), and neither does the check.
//!
//! - **The laying** is its own run of `VkCommandBuffer`s, begun to be submitted again: the
//!   encoder cut its command buffer at the outermost loop's start (`EncoderCore::cut_reusable`)
//!   and ends it at the check (`EncoderCore::end_laying`), the textures it shadows copied back at
//!   each end, behind a barrier from everything before it at its start and one to every later
//!   command and to the host at its end. A run again so repeats nothing encoded before the loop (a
//!   render pass's load and store actions, a blit, an event's signal or wait) and copies in again
//!   what it shadows; every dispatch in it is gated, and every step holds until the halted loop's
//!   copy starts.
//! - **The check** is a host step (`Deferred::Loop`), its segment the first of the batch after the
//!   laying: once the work before it completed, it reads the unit's halt word. 0: the batch goes.
//!   Else a thread of its own ([`run_again`]) submits the laying again to the command buffer's
//!   `VkQueue` with a fence of the device's, logged for the command buffer's relayed failures
//!   (`device/relay.rs`), waits for it, and reads the word again, until it is 0; then it wakes the
//!   scheduler. Meanwhile the batch waits as for a pipeline's making
//!   (`Segment::steps_after_pipelines`), and the scheduler goes on with every other line; a batch
//!   committed later on another queue that conflicts with the laying's uses waits behind it
//!   (`CommandBufferShared::footprint`), as behind any host step. The batch's other segments step
//!   once the check ends: they read what the laying leaves.
//! - **Let go.** The thread stops at the process's exit, a lost device or a failed submission or
//!   wait (each failing the command buffer as its submission would), and once the program holds
//!   no handle of the command buffer's queue (`CommandQueue::let_go`): Metal would run the loop
//!   until the process ends, but nothing of the program can then commit to the queue or wait for
//!   it, and the device would stay held. The command buffer then fails
//!   (`CommandBufferError::PredicatedLoopLetGo`), its later batches unsubmitted, and lets go of
//!   its device. Each run again is counted (`WorkCounts::predicated_loop_runs_again`).

use super::gate::Hold;
use super::{recording_error, submission_error, CommandBuffer, CommandBufferShared};
use crate::device::lock;
use crate::encoder::RecordedSegment;
use crate::error::{internal, CommandBufferError, Error};
use crate::exit_guard::{self, Call};
use crate::retained::Retained;
use crate::tracker::EncoderUses;
use crate::upload::Upload;
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long one wait for a run again's fence lasts before the thread looks for an exit.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The check after a loop's laying (the module docs).
pub(crate) struct LoopCheck {
    /// The unit's skip counter: its halt word lies 4 bytes past it.
    counter: Upload,
    /// The laying's `VkCommandBuffer`s as encoded, in order.
    raws: Vec<vk::CommandBuffer>,
    /// The command buffer's parts before the laying's first: the laying lies in the parts from
    /// there to the check's.
    from: usize,
    rounds: Mutex<Rounds>,
}

/// Where a check stands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounds {
    /// Its host step has not read the halt word yet.
    Unchecked,
    /// Its thread runs the laying again.
    Running,
    /// The loop ended, or the check stopped.
    Ended,
}

/// How one run again ended.
enum Round {
    /// The loop would run on.
    Again,
    /// The loop ended.
    Ended,
    /// The command buffer fails so.
    Failed(CommandBufferError),
    /// The process is exiting: the thread does nothing more.
    Exiting,
}

impl LoopCheck {
    /// The check of the laying in `raws`, after `from` of the command buffer's parts, of the loop
    /// of the unit of skip counter `counter`.
    pub(crate) fn new(counter: Upload, raws: Vec<vk::CommandBuffer>, from: usize) -> LoopCheck {
        LoopCheck {
            counter,
            raws,
            from,
            rounds: Mutex::new(Rounds::Unchecked),
        }
    }

    /// The command buffer's parts before the laying's first.
    pub(crate) fn from(&self) -> usize {
        self.from
    }

    /// Whether its thread runs the laying again: its batch waits.
    pub(crate) fn running(&self) -> bool {
        *lock(&self.rounds) == Rounds::Running
    }

    /// Whether the unit is halted: its halt word is not 0. Read once the work that writes it
    /// completed.
    fn halted(&self) -> bool {
        self.counter.read::<8>()[4..] != [0; 4]
    }

    /// Its host step, once the laying completed: nothing to record; where the loop halted its
    /// unit, its thread starts running the laying again. Refused where the thread cannot start.
    pub(crate) fn step(
        self: &Arc<Self>,
        command_buffer: &CommandBuffer,
    ) -> Result<RecordedSegment, Error> {
        if !self.halted() {
            *lock(&self.rounds) = Rounds::Ended;
            return Ok(nothing());
        }
        *lock(&self.rounds) = Rounds::Running;
        let check = Arc::clone(self);
        let buffer = command_buffer.0.hold();
        let spawned = std::thread::Builder::new()
            .name("metal2vulkan-loop".to_owned())
            .spawn(move || run_again(&check, buffer));
        if spawned.is_err() {
            *lock(&self.rounds) = Rounds::Ended;
            return Err(internal(
                "a predicated loop's check could not start its thread",
            ));
        }
        Ok(nothing())
    }

    /// Submit the laying again on `buffer`'s `VkQueue`, wait for it, and read the halt word.
    fn run_once(&self, buffer: &CommandBufferShared) -> Round {
        let device = &buffer.device;
        if device.is_lost() {
            return Round::Failed(CommandBufferError::DeviceLost);
        }
        let Some(busy) = exit_guard::enter() else {
            return Round::Exiting;
        };
        let shared = device.shared();
        let raw = shared.vk();
        let mut reals = Vec::with_capacity(self.raws.len());
        for &handle in &self.raws {
            // SAFETY: each ended, its command buffer committed and its batch submitted, which
            // replayed it; inside a section.
            match unsafe { crate::recorder::resolve(raw, handle) } {
                Ok(real) => reals.push(real),
                Err(error) => {
                    return Round::Failed(recording_error(device, crate::error::vulkan(error)))
                }
            }
        }
        // SAFETY: the device's fences, inside a section.
        let fence = match unsafe { shared.fences.take(raw) } {
            Ok(fence) => fence,
            Err(error) => return Round::Failed(submission_error(device, error)),
        };
        let queue = buffer.queue.vk_queue();
        let submits = [vk::SubmitInfo::default().command_buffers(&reals)];
        shared.counts.queue_submitted(1, reals.len());
        // SAFETY: the `VkQueue` is used under its lock; each command buffer was begun to be
        // submitted again, ended, and is in no pending submission: the laying's batch completed
        // before the check's step, and each run again before the next; the fence is unsignaled
        // and in no pending submission.
        let submitted = shared.submit_logged(
            (queue, None),
            Some(fence),
            &buffer.relayed,
            |vk_queue| unsafe { raw.queue_submit(vk_queue, &submits, fence) },
        );
        exit_guard::called(Call::QueueSubmit);
        if let Err(error) = submitted {
            // SAFETY: a fence no submission took.
            unsafe { crate::release::object(raw, fence) };
            return Round::Failed(submission_error(device, error));
        }
        drop(busy);
        loop {
            let Some(_busy) = exit_guard::enter() else {
                return Round::Exiting;
            };
            // SAFETY: the device's fence, submitted above.
            let waited =
                unsafe { raw.wait_for_fences(&[fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) => continue,
                // The fence stays in the relay's log, as a failed batch's does, never released.
                Err(error) => return Round::Failed(submission_error(device, error)),
            }
        }
        let Some(_busy) = exit_guard::enter() else {
            return Round::Exiting;
        };
        // Out of the relay's log before it goes back: its submission completed.
        if let Some(flights) = shared.flights(queue) {
            flights.let_go(fence);
        }
        // SAFETY: the fence is signaled and its submission completed.
        let reset = unsafe { raw.reset_fences(&[fence]) };
        exit_guard::called(Call::ResetFences);
        let mut fences = vec![fence];
        match reset {
            // SAFETY: unsignaled and in no pending submission; inside a section.
            Ok(()) => unsafe { shared.fences.give(raw, &mut fences) },
            Err(_) => {
                // SAFETY: in no pending submission.
                unsafe { crate::release::object(raw, fence) };
            }
        }
        shared.counts.predicated_loop_ran_again();
        // A failure the driver relayed for the run fails the command buffer at its completion.
        if buffer.relayed.failure().is_some() {
            return Round::Ended;
        }
        match self.halted() {
            true => Round::Again,
            false => Round::Ended,
        }
    }
}

/// A recording of nothing: a check records no command.
fn nothing() -> RecordedSegment {
    RecordedSegment {
        raws: Vec::new(),
        summary: EncoderUses::new(true).finish(),
        retained: Retained::default(),
        executions: Vec::new(),
        keys: Default::default(),
        taken: Default::default(),
        resident_misses: 0,
        faults: ReplayFaults::default(),
    }
}

/// The check's thread (the module docs): run the laying again until the loop ends, or the check
/// stops; then the scheduler takes the batch again.
fn run_again(check: &LoopCheck, buffer: Hold<CommandBufferShared>) {
    let waker = buffer.device.shared().scheduler.waker();
    let failure = loop {
        if buffer.queue.let_go() {
            break Some(CommandBufferError::PredicatedLoopLetGo);
        }
        match check.run_once(&buffer) {
            Round::Again => continue,
            Round::Ended => break None,
            Round::Failed(error) => break Some(error),
            // What it holds stays held to the process's end, as the scheduler's does.
            Round::Exiting => return std::mem::forget(buffer),
        }
    };
    if let Some(error) = failure {
        lock(&buffer.state).step_failure.get_or_insert(error);
    }
    *lock(&check.rounds) = Rounds::Ended;
    waker();
    drop(buffer);
}
