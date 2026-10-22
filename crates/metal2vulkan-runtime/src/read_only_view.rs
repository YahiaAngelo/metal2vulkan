//! Read-only views of a caller's pages: on macOS, no-copy memory made with Metal's private option
//! bit 24 ([`ResourceOptions::PRIVATE_BIT`](crate::ResourceOptions::PRIVATE_BIT)) is imported
//! through one, so that the GPU's writes to it are dropped, as Metal drops them.
//!
//! - **What Metal does.** Measured on an M3 Max under macOS 27.0.1: a no-copy buffer, or a heap
//!   over the caller's bytes, made with bit 24 takes the GPU's writes nowhere. A kernel's stores
//!   and a blit's fill reach neither the CPU nor a later command buffer, which reads the CPU's
//!   bytes; later commands of the same command buffer may read some back from the GPU's caches.
//!   The pages' protection is unchanged, and the resource's options leave the bit out.
//! - **How.** A no-copy buffer made without the bit over a read-only view of the same pages
//!   (`mach_vm_remap` sharing them, then `mprotect` read-only) behaved the same on every count
//!   measured. On macOS the device's import of a host pointer is such a buffer (MoltenVK hands
//!   the range to Metal), so the runtime imports the view in place of the caller's pointer
//!   (`Device::buffer_over_host_memory`), while the CPU reads and writes the caller's own
//!   (`Buffer::contents`). The view shares the pages, so the GPU reads what the CPU
//!   writes at the pointer.
//! - **Freed.** The view is unmapped once the memory imported from it is freed (`memory.rs`'s
//!   `Block`), as the import requires it mapped for its whole life.
//! - **Elsewhere** the import is of the caller's pages: Linux drivers pin imported pages for
//!   writing and refuse or fault on read-only ones (`host_memory.rs`), so there the GPU's writes
//!   reach the pages, and the difference is announced
//!   ([`Degradation::PrivateResourceOption`](crate::Degradation::PrivateResourceOption)).

use std::ptr::NonNull;

/// Whether this system imports no-copy memory of bit 24 through a read-only view of its pages
/// (the module docs).
pub(super) const VIEWS_DROP_WRITES: bool = cfg!(target_os = "macos");

/// A read-only view of `length` bytes of the caller's pages, mapped by the runtime and unmapped
/// as this goes: after the memory imported from it is freed (`memory.rs`'s `Block`).
#[derive(Debug)]
pub(crate) struct ReadOnlyView {
    pointer: NonNull<u8>,
    length: u64,
}

// SAFETY: the view is the runtime's alone, and nothing writes through it; unmapping it is this
// value's alone.
unsafe impl Send for ReadOnlyView {}
// SAFETY: as above; a shared view answers its pointer alone.
unsafe impl Sync for ReadOnlyView {}

impl ReadOnlyView {
    /// A read-only view of the `length` bytes of pages at `pointer`, sharing them: `None` where
    /// the system maps none.
    ///
    /// # Safety
    ///
    /// `pointer` and `length` are multiples of the host page, and the bytes are mapped memory of
    /// this process.
    #[cfg(target_os = "macos")]
    pub(super) unsafe fn of(pointer: NonNull<u8>, length: u64) -> Option<ReadOnlyView> {
        unsafe extern "C" {
            /// `<mach/mach_init.h>`: the port of this process's task.
            static mach_task_self_: libc::mach_port_t;
            fn mach_vm_remap(
                target_task: libc::vm_map_t,
                target_address: *mut libc::mach_vm_address_t,
                size: libc::mach_vm_size_t,
                mask: libc::mach_vm_address_t,
                flags: libc::c_int,
                src_task: libc::vm_map_t,
                src_address: libc::mach_vm_address_t,
                copy: libc::boolean_t,
                cur_protection: *mut libc::vm_prot_t,
                max_protection: *mut libc::vm_prot_t,
                inheritance: libc::vm_inherit_t,
            ) -> libc::kern_return_t;
        }
        // `<mach/vm_statistics.h>`: the kernel picks the address.
        const VM_FLAGS_ANYWHERE: libc::c_int = 0x0001;
        // `<mach/vm_inherit.h>`: a child process shares the view, as measured (the module docs).
        const VM_INHERIT_SHARE: libc::vm_inherit_t = 0;
        let bytes = usize::try_from(length).ok()?;
        let mut view: libc::mach_vm_address_t = 0;
        let (mut current, mut maximum) = (0, 0);
        // SAFETY: a new mapping, at an address the kernel picks, of the caller's mapped pages
        // (the caller's contract), shared rather than copied, so it reads what they hold.
        let remapped = unsafe {
            let task = mach_task_self_;
            mach_vm_remap(
                task,
                &mut view,
                length,
                0,
                VM_FLAGS_ANYWHERE,
                task,
                pointer.as_ptr() as libc::mach_vm_address_t,
                0,
                &mut current,
                &mut maximum,
                VM_INHERIT_SHARE,
            )
        };
        if remapped != libc::KERN_SUCCESS {
            return None;
        }
        let view = ReadOnlyView {
            pointer: NonNull::new(view as *mut u8)?,
            length,
        };
        // SAFETY: the mapping is the view's own, just made.
        let protected =
            unsafe { libc::mprotect(view.pointer.as_ptr().cast(), bytes, libc::PROT_READ) };
        (protected == 0).then_some(view)
    }

    /// No view is mapped on this system ([`VIEWS_DROP_WRITES`]).
    ///
    /// # Safety
    ///
    /// None: it maps nothing.
    #[cfg(not(target_os = "macos"))]
    pub(super) unsafe fn of(_pointer: NonNull<u8>, _length: u64) -> Option<ReadOnlyView> {
        None
    }

    /// Where the view starts.
    pub(super) fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }
}

impl Drop for ReadOnlyView {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: the mapping is this value's, and nothing reads it any more: the memory
        // imported from it was freed first (`memory.rs`'s `Block`).
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length as usize);
        }
    }
}
