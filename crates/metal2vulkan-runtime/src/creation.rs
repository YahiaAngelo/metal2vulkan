//! The runtime's one door to `vkCreateImage`, `vkCreateRenderPass`, `vkCreateRenderPass2` and
//! `vkCreateFramebuffer`, so their entry points stay in the exit guard's tables.
//!
//! Each is made where it is first needed: on the program's thread, in a section of the device's
//! lowering thread, which lowers render encoders' calls (`render/lowering_worker.rs`), or in a host
//! step (`exit_guard::host_step`) on the scheduler's thread, an object no earlier settle made (a
//! state made after its command buffer's commit, whose run needs an attachment's stand-in, a
//! shadow, or a pass and framebuffer of its own shape). So the four, and an image's memory
//! requirements and binding, are in the guard's table (`exit_guard::Call`) and among the calls a
//! host step makes (`exit_guard::HOST_STEP`). The device's priming (`device/prime.rs`) makes its
//! own, on the thread that makes the device. No other file calls these entry points: a source test
//! fails when one does (`tests/exit_guard.rs`).

use crate::error::{vulkan, Error};
use ash::vk;

/// A `VkImage` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateImage`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn image(
    device: &ash::Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<vk::Image, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_image(info, None) }.map_err(vulkan)
}

/// A `VkImage` of `info`, made by `device`, or the driver's refusal of it: an image whose making
/// the driver may refuse for its parameters alone (one of a DRM format modifier's explicit
/// layout, `dma_buf.rs`), whose caller takes another way then. A lost device is
/// `Error::DeviceLost`.
///
/// # Safety
///
/// As `vkCreateImage`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn image_or_refusal(
    device: &ash::Device,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<Result<vk::Image, vk::Result>, Error> {
    // SAFETY: the caller's.
    match unsafe { device.create_image(info, None) } {
        Ok(raw) => Ok(Ok(raw)),
        Err(vk::Result::ERROR_DEVICE_LOST) => Err(Error::DeviceLost),
        Err(refusal) => Ok(Err(refusal)),
    }
}

/// A `VkRenderPass` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateRenderPass`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn render_pass(
    device: &ash::Device,
    info: &vk::RenderPassCreateInfo<'_>,
) -> Result<vk::RenderPass, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_render_pass(info, None) }.map_err(vulkan)
}

/// A `VkRenderPass` of `info` (`vkCreateRenderPass2`), made by `device`.
///
/// # Safety
///
/// As `vkCreateRenderPass2`'s: `device` is live and of Vulkan 1.2 or later, and `info` and its
/// chain are valid for it.
pub(crate) unsafe fn render_pass2(
    device: &ash::Device,
    info: &vk::RenderPassCreateInfo2<'_>,
) -> Result<vk::RenderPass, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_render_pass2(info, None) }.map_err(vulkan)
}

/// A `VkFramebuffer` of `info`, made by `device`.
///
/// # Safety
///
/// As `vkCreateFramebuffer`'s: `device` is live, and `info` and its chain are valid for it.
pub(crate) unsafe fn framebuffer(
    device: &ash::Device,
    info: &vk::FramebufferCreateInfo<'_>,
) -> Result<vk::Framebuffer, Error> {
    // SAFETY: the caller's.
    unsafe { device.create_framebuffer(info, None) }.map_err(vulkan)
}
