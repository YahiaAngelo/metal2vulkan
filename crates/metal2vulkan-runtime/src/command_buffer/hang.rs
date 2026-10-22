//! A command buffer whose render encoder drew primitive type 9 directly, of 3 vertices or more,
//! at which Metal's GPU hangs (`DrawPrimitiveType::Nine`, `CommandBufferError::GpuHang`).
//! Measured on the M3 Max (macOS 27.0.1, 33 hangs, queues warmed and not): it read `Scheduled`
//! 0.68 to 13.2 ms after its commit (about 1 ms, or 4.6 ms and more: its submission's own time),
//! and Metal failed it 28.3 to 46.2 ms after its commit, 27.5 to 28.9 ms after it read
//! `Scheduled` in 26 hangs (24.2 to 37.8 in all). Every command buffer of its queue that it
//! scheduled meanwhile completed with no error together with it, in the queue's order, its event
//! signals firing: those committed at once, after the hang read `Scheduled`, 5 and 20 ms on, one
//! writing what the hang's command buffer writes, and ones whose event waits the host or another
//! queue's command buffer met during the hang. Whether their work ran is a race, which the hang's
//! submission time decides most: where it read `Scheduled` within 2 ms their fills ran on the GPU
//! during the hang in 17 of 18 command buffers, and where later in 6 of 62, the rest running
//! nothing or in part (75,648 and 80,064 bytes of a 4 MiB fill); whether the queues were warmed,
//! and whether the command buffers were made before the hang's commit, decided neither. The
//! runtime answers the later submission's outcome: `Scheduled` [`SCHEDULED`] after the commit,
//! and nothing of the others run. One whose event wait was met only after the hang completed
//! ran, as did one of another queue and one committed after.
//!
//! - **The hold.** The hanging encoder's end signals an event of the hang's own to 1 after the
//!   encoder, and adds the runtime's own wait ([`Batches::hold`]) for 2 after that, so what
//!   follows it in the command buffer (its later event signals, its completion) waits; its commit
//!   signals 3 after all its work. The signals and the wait are the runtime's, counted among no
//!   event command. The encoder and every one after it run nothing (`State::fail_from`).
//! - **`Scheduled`** with the batch before the wait, not with its last batch, which the wait
//!   holds: as that batch is submitted (or held in a group, `group.rs`) where [`SCHEDULED`] (the
//!   earliest Metal's read it) passed since the commit, else on the hang's thread once it did
//!   ([`Scheduling`]), its submission handlers first. Where a switch of storage forms rides on its
//!   commit, which settles by its last batch's submission, with its last batch.
//! - **In flight** from its commit until its work signalled 3, the queue counting it
//!   (`CommandQueue::hangs`): a thread of the hang's own makes it `Scheduled` where it is due,
//!   waits for 1 (the GPU reached the hang), lets the wait go ([`FROM_COMMIT`] after the commit,
//!   and [`FROM_REACHED`] after reaching it, the earliest Metal failed one), waits for 3, and only
//!   then stops counting it, inside a trail (`trails.rs`): a waiter of the command buffer returns
//!   once the queue no longer counts it. A command buffer that does not run its hang (its commit
//!   refused, a commit failure that fails the encoder or drops its signals, or itself discarded at
//!   its commit) sets 3 at once.
//! - **The queue's others.** A command buffer committed on the queue while a hang of it is in
//!   flight, with no event wait of its own, runs nothing and completes with no error, its event
//!   signals firing, as `commit_failed` of code 0 does ([`discarded`]); it completes after the
//!   hang, in the queue's order. One with an event wait is listed with its waits ([`behind`]);
//!   the hang's thread decides each listed one just before it lets the hold go, from its events'
//!   values then: one whose every wait is met runs nothing so, as Metal ran nothing of one it
//!   scheduled during the hang, and one whose waits are not all met runs. Its first batch's
//!   submission applies the verdict (`checks.rs`), which no batch of it can reach before the hold
//!   goes: they follow the hang's last batch in the queue's order. One committed after the
//!   decision but while the hang is still counted is decided at that submission instead, by
//!   whether the hang still is.
//!
//! [`Batches::hold`]: super::batches::Batches::hold

use super::super::event::{EventShared, SharedEvent};
use super::super::gpu_time;
use super::super::{CommandQueue, Device, Error};
use super::{lock, CommandBufferShared, CommitFailure, FailedEncoders, Item, State};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// When after its commit a command buffer its GPU hangs at reads `Scheduled`: 5.06 ms, the
/// earliest Metal's read it where its submission took 4.6 ms or more, whose queue then mostly ran
/// nothing of what it scheduled meanwhile (the module docs).
pub(super) const SCHEDULED: Duration = Duration::from_micros(5_060);

/// The earliest after its commit that Metal failed a command buffer its GPU hung at: 32.7 ms
/// (the module docs).
pub(super) const FROM_COMMIT: Duration = Duration::from_micros(32_700);

/// The earliest after it read `Scheduled` that Metal failed a command buffer its GPU hung at:
/// 27.5 ms (the module docs).
pub(super) const FROM_REACHED: Duration = Duration::from_micros(27_500);

/// How long the hang's thread waits for a signal of the GPU's (1 or 3) before it goes on as if it
/// came: a device that lost its work never signals.
const SIGNAL_BOUND: Duration = Duration::from_secs(10);

/// A queue's hangs: how many are in flight, and the command buffers committed with an event wait
/// meanwhile, not yet decided (the module docs).
#[derive(Default)]
pub(crate) struct QueueHangs {
    in_flight: AtomicUsize,
    behind: Mutex<Vec<Behind>>,
}

/// A command buffer committed with event waits while a hang of its queue was in flight: its
/// waits, and its verdict.
struct Behind {
    waits: Vec<(Arc<EventShared>, u64)>,
    verdict: Verdict,
}

/// A command buffer's verdict behind a hang ([`behind`]): undecided, it runs, or it runs nothing.
#[derive(Clone, Default)]
pub(super) struct Verdict(Arc<AtomicU8>);

impl Verdict {
    const RUNS: u8 = 1;
    const DISCARDED: u8 = 2;

    /// Whether the command buffer runs nothing, as its first batch is about to be submitted on
    /// `queue`: as decided, or where undecided, whether a hang of the queue is in flight now.
    pub(super) fn discards(&self, queue: &CommandQueue) -> bool {
        match self.0.load(Ordering::SeqCst) {
            Verdict::DISCARDED => true,
            Verdict::RUNS => false,
            _ => in_flight(queue),
        }
    }
}

impl QueueHangs {
    /// Decide every listed command buffer from its events' values now: one whose every wait is
    /// met runs nothing.
    fn decide(&self) {
        let listed: Vec<Behind> = lock(&self.behind).drain(..).collect();
        for behind in listed {
            let met = behind
                .waits
                .iter()
                .all(|(event, value)| event.value() >= *value);
            let verdict = match met {
                true => Verdict::DISCARDED,
                false => Verdict::RUNS,
            };
            behind.verdict.0.store(verdict, Ordering::SeqCst);
        }
    }
}

/// A hanging command buffer's hold: the event of its signals and its wait.
pub(super) struct Hang {
    event: SharedEvent,
    /// The batch its signal of [`Hang::REACHED`] ends: the last before its wait.
    batch: usize,
}

/// A hanging command buffer's transition to `Scheduled` (the module docs): due once
/// [`SCHEDULED`] passed since its commit, and taken once the batch before its hold
/// ([`Hang::batch`]) was submitted, or held in a group (`group.rs`); whichever comes second
/// makes it.
pub(super) struct Scheduling {
    batch: usize,
    due: bool,
    taken: bool,
}

impl Hang {
    /// The value its signal after the hanging encoder sets: the GPU reached the hang.
    pub(super) const REACHED: u64 = 1;
    /// The value its wait waits for, which its thread sets.
    pub(super) const RELEASED: u64 = 2;
    /// The value its signal after all its work sets.
    pub(super) const DONE: u64 = 3;

    /// A hold of `device`, its event at 0, whose signal of [`Hang::REACHED`] ends batch
    /// `batch`.
    pub(super) fn new(device: &Device, batch: usize) -> Result<Hang, Error> {
        Ok(Hang {
            event: device.new_shared_event()?,
            batch,
        })
    }

    /// The event of its signals and its wait.
    pub(super) fn event(&self) -> &SharedEvent {
        &self.event
    }

    /// The command buffer does not run its hang: its wait goes at once.
    pub(super) fn release(self) {
        self.event.set_signaled_value(Hang::DONE);
    }

    /// The command buffer `buffer`, of state `state`, committed at `committed`, runs its hang:
    /// its queue counts it until its work signalled [`Hang::DONE`], it reads `Scheduled`
    /// [`SCHEDULED`] after the commit or once the batch before its wait was submitted, whichever
    /// comes later ([`Scheduling`]), and its wait goes [`FROM_COMMIT`] after the commit and
    /// [`FROM_REACHED`] after the GPU reached the hang. One with a switch of storage forms riding
    /// on its commit, which settles by its last batch's submission, reads `Scheduled` with that
    /// batch. Where no thread can be made for that, the wait goes at once, the queue not counting
    /// it.
    pub(super) fn start(
        self,
        buffer: &Arc<CommandBufferShared>,
        state: &mut State,
        committed: Instant,
    ) {
        let queue = &buffer.queue;
        let hangs = Arc::clone(queue.hangs());
        let trails = Arc::clone(&queue.device().shared().trails);
        hangs.in_flight.fetch_add(1, Ordering::SeqCst);
        if state.storage_form.is_none() {
            state.hang_scheduling = Some(Scheduling {
                batch: self.batch,
                due: false,
                taken: false,
            });
        }
        let weak = Arc::downgrade(buffer);
        let event = self.event.clone();
        let builder = std::thread::Builder::new().name("metal2vulkan-hang".to_owned());
        let ended = crate::exit_guard::spawn(builder, move || {
            sleep_until(committed + SCHEDULED);
            scheduled(weak);
            event.wait_until_signaled_value(Hang::REACHED, SIGNAL_BOUND);
            sleep_until((committed + FROM_COMMIT).max(Instant::now() + FROM_REACHED));
            let trail = trails.open();
            hangs.decide();
            event.set_signaled_value(Hang::RELEASED);
            event.wait_until_signaled_value(Hang::DONE, SIGNAL_BOUND);
            hangs.in_flight.fetch_sub(1, Ordering::SeqCst);
            drop(event);
            drop(hangs);
            drop(trail);
        });
        if ended.is_err() {
            state.hang_scheduling = None;
            queue.hangs().decide();
            queue.hangs().in_flight.fetch_sub(1, Ordering::SeqCst);
            self.event.set_signaled_value(Hang::DONE);
        }
    }
}

/// Sleep until `until`, if it is still ahead.
fn sleep_until(until: Instant) {
    if let Some(left) = until.checked_duration_since(Instant::now()) {
        std::thread::sleep(left);
    }
}

/// The hanging command buffer `buffer` holds, where something still holds it, came to its time to
/// read `Scheduled` ([`Scheduling`]): it does here where the batch before its wait was taken
/// (its kernel end then, and its submission handlers on this thread first), and else as that
/// batch is ([`schedules`]). This thread never lets go of the command buffer's last reference,
/// which holds the device: that goes in a context of its own (`Device::let_go_elsewhere`).
fn scheduled(buffer: Weak<CommandBufferShared>) {
    let Some(buffer) = buffer.upgrade() else {
        return;
    };
    let made = {
        let mut state = lock(&buffer.state);
        let taken = match &mut state.hang_scheduling {
            Some(scheduling) => {
                scheduling.due = true;
                scheduling.taken
            }
            None => false,
        };
        let handlers = taken && !state.submission.is_empty();
        if taken {
            state.submitting = handlers;
        }
        taken.then_some(handlers)
    };
    if let Some(handlers) = made {
        buffer.kernel.end_at(gpu_time::host_nanos());
        buffer.schedule();
        if handlers {
            buffer.run_submission_handlers();
        }
    }
    if let Some(last) = Arc::into_inner(buffer) {
        let (contexts, context) = last.device.elsewhere();
        contexts.post(context, Item::Release(Box::new(last)));
    }
}

/// Whether batch `index` of a command buffer, its last where `last`, makes it `Scheduled` as it
/// is submitted or held in a group: its last, or, behind its GPU hang (`scheduling`), the batch
/// before the hang's wait where its time to read `Scheduled` came; no other ([`Scheduling`]).
pub(super) fn schedules(scheduling: &mut Option<Scheduling>, index: usize, last: bool) -> bool {
    match scheduling {
        None => last,
        Some(scheduling) if index == scheduling.batch => {
            scheduling.taken = true;
            scheduling.due
        }
        Some(_) => false,
    }
}

/// Whether a hang of `queue` is in flight (the module docs).
pub(super) fn in_flight(queue: &CommandQueue) -> bool {
    queue.hangs().in_flight.load(Ordering::SeqCst) > 0
}

/// The verdict of a command buffer committed on `queue` with the event waits `waits` gives,
/// listed for the hang's decision where a hang of the queue is in flight; `None` where none is
/// (the module docs).
pub(super) fn behind(
    queue: &CommandQueue,
    waits: impl FnOnce() -> Vec<(Arc<EventShared>, u64)>,
) -> Option<Verdict> {
    if !in_flight(queue) {
        return None;
    }
    let verdict = Verdict::default();
    lock(&queue.hangs().behind).push(Behind {
        waits: waits(),
        verdict: verdict.clone(),
    });
    Some(verdict)
}

/// How a command buffer committed on `queue` with `event_waits` event waits of its own fails
/// where a hang of the queue is in flight: it runs nothing and completes with no error, its
/// signals firing (the module docs). `None` where none is, or where it waits for an event, which
/// its first batch's submission decides.
pub(super) fn discarded(queue: &CommandQueue, event_waits: usize) -> Option<CommitFailure> {
    (in_flight(queue) && event_waits == 0).then_some(CommitFailure {
        code: 0,
        encoders: FailedEncoders::From(0),
        signals: true,
    })
}
