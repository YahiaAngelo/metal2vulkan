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
//! - **`Scheduled`** [`SCHEDULED`] (the earliest Metal's read it) after the commit, not with its
//!   last batch, which the wait holds: the hang's thread makes it then, once the batch before the
//!   wait was taken, submitted or held in a group (`group.rs`), its submission handlers first
//!   ([`Scheduling::Hang`]). Where that batch is not taken within [`SIGNAL_BOUND`] (an event wait
//!   before the hang not met), or a switch of storage forms rides on its commit, which settles by
//!   its last batch's submission, it reads `Scheduled` with its last batch.
//! - **In flight** from its commit until its work signalled 3, the queue counting it
//!   (`CommandQueue::hangs`): a thread of the hang's own makes it `Scheduled`, waits for 1 (the
//!   GPU reached the hang), makes the queue's command buffers committed meanwhile `Scheduled`
//!   (below) until it lets the wait go ([`FROM_COMMIT`] after the commit, and [`FROM_REACHED`]
//!   after reaching it, the earliest Metal failed one), waits for 3, and only then stops counting
//!   it, inside a trail (`trails.rs`): a waiter of the command buffer returns once the queue no
//!   longer counts it. A command buffer that does not run its hang (its commit refused, a commit
//!   failure that fails the encoder or drops its signals, or itself discarded at its commit) sets
//!   3 at once.
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
//! - **Their `Scheduled`**, as Metal's: from the hang's on, as they are committed, each once its
//!   event waits are met. Each joins the hang's list at its commit ([`join`]); the hang's thread,
//!   once it made the hang `Scheduled`, makes each listed one `Scheduled` in the queue's order,
//!   once its waits are met, its submission handlers first and its kernel times then
//!   ([`QueueHangs::schedule_early`], [`Scheduling::Behind`]), until it lets the wait go. One
//!   whose waits are not met by then, and every one after it, read `Scheduled` with their last
//!   batch, after the hang's, as does every one where the hang's own came with its last batch.
//!
//! [`Batches::hold`]: super::batches::Batches::hold

use super::super::event::{EventShared, SharedEvent};
use super::super::exit_guard;
use super::super::gpu_time;
use super::super::{CommandQueue, Device, Error};
use super::{lock, CommandBufferShared, CommitFailure, FailedEncoders, Item, State};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
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

/// A queue's hangs: how many are in flight, the command buffers committed with an event wait
/// meanwhile, not yet decided, and those committed meanwhile, not yet `Scheduled` (the module
/// docs).
#[derive(Default)]
pub(crate) struct QueueHangs {
    in_flight: AtomicUsize,
    behind: Mutex<Vec<Behind>>,
    early: Mutex<Early>,
    /// Signaled as a command buffer joins `early`.
    joined: Condvar,
}

/// The command buffers committed on a queue while a hang of it is in flight, in their order, for
/// the hang's thread to make `Scheduled` early (the module docs); each of the hang it joined
/// under, which the hang's start counts.
#[derive(Default)]
struct Early {
    hang: u64,
    joined: VecDeque<Joined>,
}

/// A command buffer that joined [`Early`]: the hang it joined under, the command buffer, and its
/// event waits.
struct Joined {
    hang: u64,
    buffer: Weak<CommandBufferShared>,
    waits: Vec<(Arc<EventShared>, u64)>,
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
    /// Until `until`, make the command buffers that joined [`Early`] under hang `hang`
    /// `Scheduled`, in their order, each once its event waits are met (the module docs). One
    /// whose waits are not met by then, and every one after it, read `Scheduled` with their last
    /// batch.
    fn schedule_early(&self, hang: u64, until: Instant) {
        loop {
            let next = {
                let mut early = lock(&self.early);
                loop {
                    if let Some(joined) = early.joined.pop_front() {
                        break Some(joined);
                    }
                    let left = until.saturating_duration_since(Instant::now());
                    if left.is_zero() || exit_guard::closed() {
                        break None;
                    }
                    early = (self.joined.wait_timeout(early, left))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            };
            let Some(joined) = next else {
                return;
            };
            if joined.hang != hang {
                continue;
            }
            let met = (joined.waits.iter()).all(|(event, value)| {
                event.wait_until(*value, until.saturating_duration_since(Instant::now()))
            });
            if !met {
                sleep_until(until);
                return;
            }
            behind_scheduled(joined.buffer);
        }
    }

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

/// A command buffer's transition to `Scheduled` around a hang of its queue (the module docs).
pub(super) enum Scheduling {
    /// The hang's own: its thread makes it [`SCHEDULED`] after the commit, once the batch before
    /// its wait (`batch`, [`Hang::batch`]) was taken, submitted or held in a group (`group.rs`).
    Hang { batch: usize, taken: bool },
    /// One committed while a hang of its queue was in flight: the hang's thread makes it early,
    /// or its last batch does, whichever comes first.
    Behind { made: bool },
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
    /// its queue counts it until its work signalled [`Hang::DONE`]; it reads `Scheduled`
    /// [`SCHEDULED`] after the commit, or later once the batch before its wait was taken
    /// ([`Scheduling::Hang`]), and the queue's command buffers committed meanwhile follow
    /// ([`QueueHangs::schedule_early`]); its wait goes [`FROM_COMMIT`] after the commit and
    /// [`FROM_REACHED`] after the GPU reached the hang. One with a switch of storage forms riding
    /// on its commit, which settles by its last batch's submission, reads `Scheduled` with that
    /// batch, as do the queue's others. Where no thread can be made for that, the wait goes at
    /// once, the queue not counting it.
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
        let hang = {
            let mut early = lock(&hangs.early);
            early.hang += 1;
            early.hang
        };
        if state.storage_form.is_none() {
            state.hang_scheduling = Some(Scheduling::Hang {
                batch: self.batch,
                taken: false,
            });
        }
        let weak = Arc::downgrade(buffer);
        let event = self.event.clone();
        let builder = std::thread::Builder::new().name("metal2vulkan-hang".to_owned());
        let ended = crate::exit_guard::spawn(builder, move || {
            sleep_until(committed + SCHEDULED);
            let scheduled = hang_scheduled(weak);
            event.wait_until_signaled_value(Hang::REACHED, SIGNAL_BOUND);
            let release = (committed + FROM_COMMIT).max(Instant::now() + FROM_REACHED);
            match scheduled {
                true => hangs.schedule_early(hang, release),
                false => sleep_until(release),
            }
            let trail = trails.open();
            hangs.decide();
            event.set_signaled_value(Hang::RELEASED);
            event.wait_until_signaled_value(Hang::DONE, SIGNAL_BOUND);
            lock(&hangs.early).joined.clear();
            hangs.in_flight.fetch_sub(1, Ordering::SeqCst);
            drop(event);
            drop(hangs);
            drop(trail);
        });
        if ended.is_err() {
            state.hang_scheduling = None;
            queue.hangs().decide();
            lock(&queue.hangs().early).joined.clear();
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
/// read `Scheduled` ([`Scheduling::Hang`]): it does once the batch before its wait was taken, its
/// submission handlers first on this thread. Where that batch is not taken within
/// [`SIGNAL_BOUND`] (an event wait before the hang not met), its last batch makes it instead.
/// Whether it was made here.
fn hang_scheduled(buffer: Weak<CommandBufferShared>) -> bool {
    let Some(buffer) = buffer.upgrade() else {
        return false;
    };
    let deadline = Instant::now() + SIGNAL_BOUND;
    let made = {
        let mut state = lock(&buffer.state);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match state.hang_scheduling {
                Some(Scheduling::Hang { taken: true, .. }) => {
                    let handlers = !state.submission.is_empty();
                    state.submitting = handlers;
                    break Some(handlers);
                }
                Some(Scheduling::Hang { taken: false, .. })
                    if !left.is_zero() && !exit_guard::closed() => {}
                Some(Scheduling::Hang { taken: false, .. }) => {
                    state.hang_scheduling = None;
                    break None;
                }
                _ => break None,
            }
            state.parked += 1;
            state = (buffer
                .changed
                .wait_timeout(state, left.min(exit_guard::WAKE)))
            .unwrap_or_else(PoisonError::into_inner)
            .0;
            state.parked -= 1;
        }
    };
    if let Some(handlers) = made {
        transition(&buffer, handlers);
    }
    let_go(buffer);
    made.is_some()
}

/// The command buffer `buffer` holds, where something still holds it, committed while a hang of
/// its queue was in flight, reads `Scheduled` here, where its last batch has not made it
/// ([`Scheduling::Behind`]): its kernel times then, its submission handlers first on this
/// thread.
fn behind_scheduled(buffer: Weak<CommandBufferShared>) {
    let Some(buffer) = buffer.upgrade() else {
        return;
    };
    let made = {
        let mut state = lock(&buffer.state);
        match &mut state.hang_scheduling {
            Some(Scheduling::Behind { made }) if !*made => {
                *made = true;
                let handlers = !state.submission.is_empty();
                state.submitting = handlers;
                Some(handlers)
            }
            _ => None,
        }
    };
    if let Some(handlers) = made {
        buffer.kernel.begin();
        transition(&buffer, handlers);
    }
    let_go(buffer);
}

/// `buffer` becomes `Scheduled` here, its kernel end now: its transition, then its submission
/// handlers on this thread, which the transition waits for where `handlers` (the caller set
/// `submitting` under the state's lock).
fn transition(buffer: &Arc<CommandBufferShared>, handlers: bool) {
    buffer.kernel.end_at(gpu_time::host_nanos());
    buffer.schedule();
    if handlers {
        buffer.run_submission_handlers();
    }
}

/// Let go of `buffer` here, unless this is its last reference, which holds the device: that goes
/// in a context of its own (`Device::let_go_elsewhere`), so the hang's thread never runs the
/// device's teardown.
fn let_go(buffer: Arc<CommandBufferShared>) {
    if let Some(last) = Arc::into_inner(buffer) {
        let (contexts, context) = last.device.elsewhere();
        contexts.post(context, Item::Release(Box::new(last)));
    }
}

/// Whether batch `index` of a command buffer, its last where `last`, makes it `Scheduled` as it
/// is submitted or held in a group ([`Scheduling`]): its last, but where its hang's thread makes
/// it or the hang's thread made it early; and whether it is the hang's batch before its wait,
/// taken now, which the hang's thread waits for (on the command buffer's `changed`).
pub(super) fn schedules(
    scheduling: &mut Option<Scheduling>,
    index: usize,
    last: bool,
) -> (bool, bool) {
    match scheduling {
        None => (last, false),
        Some(Scheduling::Hang { batch, taken }) => {
            let took = index == *batch && !*taken;
            *taken |= took;
            (false, took)
        }
        Some(Scheduling::Behind { made }) => {
            let makes = last && !*made;
            *made |= makes;
            (makes, false)
        }
    }
}

/// The command buffer `buffer`, of state `state`, committed on its queue while a hang of it is in
/// flight with the event waits `waits`, joins the hang's command buffers to make `Scheduled`
/// early ([`QueueHangs::schedule_early`]).
pub(super) fn join(
    buffer: &Arc<CommandBufferShared>,
    state: &mut State,
    waits: Vec<(Arc<EventShared>, u64)>,
) {
    state.hang_scheduling = Some(Scheduling::Behind { made: false });
    let hangs = buffer.queue.hangs();
    let mut early = lock(&hangs.early);
    let hang = early.hang;
    early.joined.push_back(Joined {
        hang,
        buffer: Arc::downgrade(buffer),
        waits,
    });
    drop(early);
    hangs.joined.notify_all();
}

/// Whether a hang of `queue` is in flight (the module docs).
pub(super) fn in_flight(queue: &CommandQueue) -> bool {
    queue.hangs().in_flight.load(Ordering::SeqCst) > 0
}

/// The verdict of a command buffer committed on `queue` with the event waits `waits` gives,
/// listed for the hang's decision where a hang of the queue is in flight; `None` where none is
/// (the module docs).
pub(super) fn behind(queue: &CommandQueue, waits: &[(Arc<EventShared>, u64)]) -> Option<Verdict> {
    if !in_flight(queue) {
        return None;
    }
    let verdict = Verdict::default();
    lock(&queue.hangs().behind).push(Behind {
        waits: waits.to_vec(),
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
