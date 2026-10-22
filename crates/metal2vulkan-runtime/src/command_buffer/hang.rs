//! A command buffer whose render encoder drew primitive type 9 directly, of 3 vertices or more,
//! at which Metal's GPU hangs (`DrawPrimitiveType::Nine`, `CommandBufferError::GpuHang`).
//! Measured on the M3 Max (macOS 27.0.1, 7 hangs, queues warmed and not): Metal failed it 32.7
//! to 42.7 ms after its commit and 27.5 to 29.5 ms after it read `Scheduled`, and ran nothing of
//! any command buffer of its queue that it scheduled meanwhile, those committed at once, after the
//! hang read `Scheduled`, 5 and 20 ms on, one writing what the hang's command buffer writes, and
//! ones whose event waits the host or another queue's command buffer met during the hang: each
//! completed with no error together with it, its event signals firing. One whose event wait was
//! met only after the hang completed ran, as did one of another queue and one committed after.
//!
//! - **The hold.** The hanging encoder's end signals an event of the hang's own to 1 after the
//!   encoder, and adds the runtime's own wait ([`Batches::hold`]) for 2 after that, so what
//!   follows it in the command buffer (its later event signals, its completion) waits; its commit
//!   signals 3 after all its work. The signals and the wait are the runtime's, counted among no
//!   event command. The encoder and every one after it run nothing (`State::fail_from`).
//! - **In flight** from its commit until its work signalled 3, the queue counting it
//!   (`CommandQueue::hangs`): a thread of the hang's own waits for 1 (the GPU reached the
//!   hang), lets the wait go ([`FROM_COMMIT`] after the commit, and [`FROM_REACHED`] after
//!   reaching it, the earliest Metal failed one), waits for 3, and only then stops counting it,
//!   inside a trail (`trails.rs`): a waiter of the command buffer returns once the queue no longer
//!   counts it. A command buffer that does not run its hang (its commit refused, a commit failure
//!   that fails the encoder or drops its signals, or itself discarded at its commit) sets 3 at
//!   once.
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
use super::super::{CommandQueue, Device, Error};
use super::{lock, CommitFailure, FailedEncoders};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
}

impl Hang {
    /// The value its signal after the hanging encoder sets: the GPU reached the hang.
    pub(super) const REACHED: u64 = 1;
    /// The value its wait waits for, which its thread sets.
    pub(super) const RELEASED: u64 = 2;
    /// The value its signal after all its work sets.
    pub(super) const DONE: u64 = 3;

    /// A hold of `device`, its event at 0.
    pub(super) fn new(device: &Device) -> Result<Hang, Error> {
        Ok(Hang {
            event: device.new_shared_event()?,
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

    /// The command buffer committed at `committed` on `queue` runs its hang: the queue counts it
    /// until its work signalled [`Hang::DONE`], and its wait goes [`FROM_COMMIT`] after the
    /// commit and [`FROM_REACHED`] after the GPU reached the hang. Where no thread can be made for
    /// that, the wait goes at once, the queue not counting it.
    pub(super) fn start(self, queue: &CommandQueue, committed: Instant) {
        let hangs = Arc::clone(queue.hangs());
        let trails = Arc::clone(&queue.device().shared().trails);
        hangs.in_flight.fetch_add(1, Ordering::SeqCst);
        let event = self.event.clone();
        let builder = std::thread::Builder::new().name("metal2vulkan-hang".to_owned());
        let ended = crate::exit_guard::spawn(builder, move || {
            event.wait_until_signaled_value(Hang::REACHED, SIGNAL_BOUND);
            let until = (committed + FROM_COMMIT).max(Instant::now() + FROM_REACHED);
            if let Some(left) = until.checked_duration_since(Instant::now()) {
                std::thread::sleep(left);
            }
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
            queue.hangs().decide();
            queue.hangs().in_flight.fetch_sub(1, Ordering::SeqCst);
            self.event.set_signaled_value(Hang::DONE);
        }
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
