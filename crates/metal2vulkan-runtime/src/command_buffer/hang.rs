//! A command buffer whose render encoder drew primitive type 9 directly, of 3 vertices or more,
//! at which Metal's GPU hangs (`DrawPrimitiveType::Nine`, `CommandBufferError::GpuHang`):
//! measured on the M3 Max (macOS 27.0.1, 4 hangs), Metal failed it 37.7 to 42.7 ms after its
//! commit, and ran nothing of the command buffers of its queue committed meanwhile, which
//! completed with no error after it, their event signals firing; one whose event wait held it
//! until the hang ended ran, as did one of another queue and one committed after it.
//!
//! - **The hold.** The hanging encoder's end adds the runtime's own wait ([`Batches::hold`]) for
//!   an event of the command buffer's own after the encoder, so what follows it in the command
//!   buffer (its later event signals, its completion) waits; the encoder and every one after it
//!   run nothing (`State::fail_from`).
//! - **In flight** from its commit until [`HANG_WINDOW`] on (the earliest Metal failed one), when
//!   a thread of the hang's own lets the wait go: meanwhile the queue counts it
//!   (`CommandQueue::hangs`). A command buffer that does not run (its commit refused, or itself
//!   committed while another hang was in flight) lets its wait go at once.
//! - **The queue's others.** A command buffer committed on the queue while a hang of it is in
//!   flight, with no event wait of its own, runs nothing and completes with no error, its event
//!   signals firing, as `commit_failed` of code 0 does; it completes after the hang, in the
//!   queue's order. One with an event wait runs, as Metal ran one whose wait held it past the
//!   hang (Metal's answer where the wait is met during the hang is unmeasured).
//!
//! [`Batches::hold`]: super::batches::Batches::hold

use super::super::event::SharedEvent;
use super::super::{CommandQueue, Device, Error};
use super::{CommitFailure, FailedEncoders};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// How long after its commit a command buffer Metal's GPU hangs at stays in flight: Metal failed
/// one 37.7 to 42.7 ms after its commit (the module docs).
pub(super) const HANG_WINDOW: Duration = Duration::from_millis(37);

/// A hanging command buffer's hold: the event its own wait waits for.
pub(super) struct Hang {
    event: SharedEvent,
}

impl Hang {
    /// A hold of `device`, its event at 0.
    pub(super) fn new(device: &Device) -> Result<Hang, Error> {
        Ok(Hang {
            event: device.new_shared_event()?,
        })
    }

    /// The event the hold's wait waits for, to reach 1.
    pub(super) fn event(&self) -> &SharedEvent {
        &self.event
    }

    /// The command buffer runs nothing of it: the wait goes at once.
    pub(super) fn release(self) {
        self.event.set_signaled_value(1);
    }

    /// The command buffer committed at `committed` on `queue` and runs: its queue counts the
    /// hang until [`HANG_WINDOW`] after the commit, when the wait goes. Where no thread can be
    /// made for that, the wait goes at once, the queue not counting it.
    pub(super) fn start(self, queue: &CommandQueue, committed: Instant) {
        let hangs = std::sync::Arc::clone(queue.hangs());
        hangs.fetch_add(1, Ordering::SeqCst);
        let event = self.event.clone();
        let ended = std::thread::Builder::new()
            .name("metal2vulkan-hang".to_owned())
            .spawn(move || {
                let until = committed + HANG_WINDOW;
                if let Some(left) = until.checked_duration_since(Instant::now()) {
                    std::thread::sleep(left);
                }
                hangs.fetch_sub(1, Ordering::SeqCst);
                event.set_signaled_value(1);
            });
        if ended.is_err() {
            queue.hangs().fetch_sub(1, Ordering::SeqCst);
            self.event.set_signaled_value(1);
        }
    }
}

/// How a command buffer committed on `queue` with `event_waits` event waits of its own fails
/// where a hang of the queue is in flight: it runs nothing and completes with no error, its
/// signals firing (the module docs). `None` where none is, or where it waits for an event.
pub(super) fn discarded(queue: &CommandQueue, event_waits: usize) -> Option<CommitFailure> {
    let in_flight = queue.hangs().load(Ordering::SeqCst) > 0;
    (in_flight && event_waits == 0).then_some(CommitFailure {
        code: 0,
        encoders: FailedEncoders::From(0),
        signals: true,
    })
}
