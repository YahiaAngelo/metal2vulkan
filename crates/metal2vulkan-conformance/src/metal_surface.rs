//! The IOSurfaces the Metal side lays a plane texture over, on macOS only.
//!
//! A texture of a two-plane YCbCr format exists **only** over an IOSurface:
//! `newTextureWithDescriptor:` answers nil for every one of the 69 formats, and so does a buffer's
//! texture of them (`cases::ycbcr_textures::FormatDescriptors` is that fact, recorded). So the
//! Metal side cannot make one from a descriptor the way it makes every other texture, and until
//! this module existed it answered `Failure::unavailable` and whole families carried their Metal
//! answer from a probe instead of being recorded.
//!
//! What a case states is a **buffer standing for the surface**, with each plane's first byte and
//! `bytesPerRow` ([`crate::case::PlaneTexture`]): that is what lets the runtime side run the case
//! where there is no CoreVideo. This module makes a real IOSurface whose allocation is laid out
//! exactly as the case says the buffer is — same plane offsets, same row pitches, same length —
//! and copies the buffer's bytes into it from byte 0. The two sides then read the same bytes
//! through the same layout, which is what makes their answers comparable at all.
//!
//! # The layout is asked for, and then checked
//!
//! `IOSurfaceCreate` is free to lay a surface out its own way: it reads the plane dictionary as a
//! request. So [`Surface::two_plane`] states the layout and [`Surface::planes_are_as_asked`] reads
//! it back plane by plane, and a surface that came out laid differently is a typed failure rather
//! than a comparison against bytes in other places. Nothing here silently repitches: a case's
//! recording says where the planes are, and if Apple's allocator disagrees, the case must be told
//! so rather than quietly answered.
//!
//! # The geometry comes from the format, not from a name table
//!
//! A plane's element size and its size in texels follow from the format's own layout —
//! `ycbcr_format`'s bits and chroma subsampling ([`planes_of`]) — not from the surface's
//! four-character OSType. `core_video_planes` is the guard on that derivation, used by the tests
//! alone: CoreVideo lays out a buffer of the same OSType and size, and the derivation must agree
//! with Apple's own answer plane for plane.
//!
//! The frameworks are reached through their C functions (CoreFoundation and IOSurface), as
//! `metal2vulkan-validation`'s own surface probe reaches them; each object made here is released
//! here.

use core::ffi::c_void;
use core::ptr::NonNull;
use metal2vulkan::metal::format::{ycbcr_format, YCbCrFormat};
use metal2vulkan::metal::PixelFormat;
use objc2::encode::{Encoding, RefEncode};

/// `IOSurfaceRef`'s pointee, opaque, encoded as Metal's methods declare it.
#[repr(C)]
pub struct IoSurfaceOpaque {
    _private: [u8; 0],
}

// SAFETY: `IOSurfaceRef` is `struct __IOSurface *`, which Objective-C encodes so.
unsafe impl RefEncode for IoSurfaceOpaque {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("__IOSurface", &[]));
}

type CfType = *const c_void;
type CfAllocator = *const c_void;
type CfMutableDictionary = *mut c_void;

/// `kCFNumberSInt64Type`.
const NUMBER_SINT64: isize = 4;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    static kCFTypeDictionaryKeyCallBacks: u8;
    static kCFTypeDictionaryValueCallBacks: u8;
    static kCFTypeArrayCallBacks: u8;
    fn CFDictionaryCreateMutable(
        allocator: CfAllocator,
        capacity: isize,
        keys: *const u8,
        values: *const u8,
    ) -> CfMutableDictionary;
    fn CFDictionarySetValue(dictionary: CfMutableDictionary, key: CfType, value: CfType);
    fn CFNumberCreate(allocator: CfAllocator, number_type: isize, value: *const c_void) -> CfType;
    fn CFArrayCreate(
        allocator: CfAllocator,
        values: *const CfType,
        count: isize,
        callbacks: *const u8,
    ) -> CfType;
    fn CFRelease(object: CfType);
}

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    static kIOSurfaceWidth: CfType;
    static kIOSurfaceHeight: CfType;
    static kIOSurfaceBytesPerElement: CfType;
    static kIOSurfacePixelFormat: CfType;
    static kIOSurfaceAllocSize: CfType;
    static kIOSurfacePlaneInfo: CfType;
    static kIOSurfacePlaneWidth: CfType;
    static kIOSurfacePlaneHeight: CfType;
    static kIOSurfacePlaneBytesPerRow: CfType;
    static kIOSurfacePlaneOffset: CfType;
    static kIOSurfacePlaneSize: CfType;
    static kIOSurfacePlaneBytesPerElement: CfType;
    static kIOSurfacePlaneElementWidth: CfType;
    static kIOSurfacePlaneElementHeight: CfType;
    fn IOSurfaceCreate(properties: CfType) -> *mut IoSurfaceOpaque;
    fn IOSurfaceLock(surface: *mut IoSurfaceOpaque, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut IoSurfaceOpaque, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(surface: *mut IoSurfaceOpaque) -> *mut c_void;
    fn IOSurfaceGetBaseAddressOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: *mut IoSurfaceOpaque) -> usize;
    fn IOSurfaceGetPlaneCount(surface: *mut IoSurfaceOpaque) -> usize;
    fn IOSurfaceGetWidthOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerRowOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
    fn IOSurfaceGetBytesPerElementOfPlane(surface: *mut IoSurfaceOpaque, plane: usize) -> usize;
}

/// One plane of a surface: where it lies in the allocation, and how its elements are shaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    /// The plane's size in texels of its own (a chroma plane of a 4:2:0 format is half the
    /// texture's, each way).
    pub width: u64,
    pub height: u64,
    /// The plane's first byte in the allocation.
    pub offset: u64,
    /// The bytes between its rows.
    pub bytes_per_row: u64,
    /// The bytes of one of its elements: a luma code, or a chroma pair.
    pub bytes_per_element: u64,
}

impl PlaneLayout {
    /// The bytes of the plane that hold texels: its elements, row by row. Whatever the pitch
    /// leaves over is the surface's own padding and holds nothing a texel reads.
    fn texel_bytes_per_row(&self) -> u64 {
        self.width * self.bytes_per_element
    }
}

/// The planes a two-plane YCbCr texture of `format` over a surface of `size` texels has, with
/// each plane's first byte and row pitch taken from `at` (the case's own statement of where the
/// planes lie).
///
/// The geometry is the format's: a code is one byte at 8 bits and two at 10 or 12, a chroma
/// element is a Cb and a Cr code, and the chroma plane is halved along each axis the format's
/// subsampling halves. A packed format (`_PACKED`, whose codes do not sit one to a byte or one to
/// a 16-bit word) is refused here rather than guessed at.
pub fn planes_of(
    format: PixelFormat,
    size: [u32; 2],
    at: [(u64, u64); 2],
) -> Result<[PlaneLayout; 2], String> {
    let YCbCrFormat {
        bits,
        subsampling,
        packed,
        ..
    } = ycbcr_format(format)
        .ok_or_else(|| format!("{format:?} is not a two-plane YCbCr format"))?;
    if packed {
        return Err(format!(
            "{format:?} packs its codes, and this side lays out only the forms whose codes sit one \
             to a byte or one to a 16-bit word"
        ));
    }
    let code = match bits {
        8 => 1,
        10 | 12 => 2,
        other => return Err(format!("{format:?} is {other} bits a code")),
    };
    let [halve_x, halve_y] = subsampling.halved();
    let chroma = |extent: u32, halve: bool| {
        let extent = u64::from(extent);
        if halve {
            extent.div_ceil(2)
        } else {
            extent
        }
    };
    Ok([
        PlaneLayout {
            width: u64::from(size[0]),
            height: u64::from(size[1]),
            offset: at[0].0,
            bytes_per_row: at[0].1,
            bytes_per_element: code,
        },
        PlaneLayout {
            width: chroma(size[0], halve_x),
            height: chroma(size[1], halve_y),
            offset: at[1].0,
            bytes_per_row: at[1].1,
            // A chroma element is the pair: a Cb code and a Cr code.
            bytes_per_element: 2 * code,
        },
    ])
}

/// A CoreFoundation object made here, released as it goes.
struct Owned(CfType);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: an object this module made, released once.
            unsafe { CFRelease(self.0) };
        }
    }
}

fn number(value: u64) -> Owned {
    let value = value as i64;
    // SAFETY: a 64-bit integer read through its pointer during the call.
    Owned(unsafe {
        CFNumberCreate(
            core::ptr::null(),
            NUMBER_SINT64,
            (&raw const value).cast::<c_void>(),
        )
    })
}

/// A mutable dictionary of `pairs`, its keys IOSurface's string constants.
fn dictionary(pairs: &[(CfType, CfType)]) -> Owned {
    // SAFETY: CoreFoundation's own callbacks; each key and value is a live CF object, which the
    // dictionary retains.
    unsafe {
        let made = CFDictionaryCreateMutable(
            core::ptr::null(),
            0,
            &raw const kCFTypeDictionaryKeyCallBacks,
            &raw const kCFTypeDictionaryValueCallBacks,
        );
        for &(key, value) in pairs {
            CFDictionarySetValue(made, key, value);
        }
        Owned(made.cast_const())
    }
}

fn array(values: &[CfType]) -> Owned {
    // SAFETY: live CF objects, which the array retains.
    Owned(unsafe {
        CFArrayCreate(
            core::ptr::null(),
            values.as_ptr(),
            values.len() as isize,
            &raw const kCFTypeArrayCallBacks,
        )
    })
}

/// A surface made here, released as it goes.
pub struct Surface(NonNull<IoSurfaceOpaque>);

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: the surface `IOSurfaceCreate` made, released once.
        unsafe { CFRelease(self.0.as_ptr().cast_const().cast::<c_void>()) };
    }
}

impl Surface {
    /// A surface of `size` texels and OSType `ostype`, with `planes` where they say they are, in
    /// an allocation of `length` bytes.
    pub fn two_plane(
        ostype: [u8; 4],
        size: [u32; 2],
        planes: &[PlaneLayout; 2],
        length: usize,
    ) -> Result<Surface, String> {
        // SAFETY: IOSurface's key constants, read once each.
        let plane_keys = unsafe {
            [
                kIOSurfacePlaneWidth,
                kIOSurfacePlaneHeight,
                kIOSurfacePlaneBytesPerRow,
                kIOSurfacePlaneOffset,
                kIOSurfacePlaneSize,
                kIOSurfacePlaneBytesPerElement,
                kIOSurfacePlaneElementWidth,
                kIOSurfacePlaneElementHeight,
            ]
        };
        // The numbers are held until the surface is made: the dictionaries retain them, and the
        // array retains the dictionaries, so nothing here may be dropped early.
        let mut held = Vec::new();
        let mut infos = Vec::new();
        for plane in planes {
            let values = [
                plane.width,
                plane.height,
                plane.bytes_per_row,
                plane.offset,
                plane.bytes_per_row * plane.height,
                plane.bytes_per_element,
                // One element to a texel, each way: the subsampling is in the plane's own size,
                // which is how CoreVideo states a two-plane surface.
                1,
                1,
            ]
            .map(number);
            let pairs: Vec<_> = plane_keys
                .iter()
                .zip(&values)
                .map(|(&key, value)| (key, value.0))
                .collect();
            let info = dictionary(&pairs);
            infos.push(info.0);
            held.push((info, values));
        }
        let info = array(&infos);
        let numbers = [
            u64::from(size[0]),
            u64::from(size[1]),
            planes[0].bytes_per_element,
            u64::from(u32::from_be_bytes(ostype)),
            length as u64,
        ]
        .map(number);
        // SAFETY: IOSurface's key constants, read once each.
        let keys = unsafe {
            [
                kIOSurfaceWidth,
                kIOSurfaceHeight,
                kIOSurfaceBytesPerElement,
                kIOSurfacePixelFormat,
                kIOSurfaceAllocSize,
            ]
        };
        let mut pairs: Vec<_> = keys
            .iter()
            .zip(&numbers)
            .map(|(&key, value)| (key, value.0))
            .collect();
        // SAFETY: IOSurface's key constant, read once.
        pairs.push((unsafe { kIOSurfacePlaneInfo }, info.0));
        let properties = dictionary(&pairs);
        // SAFETY: a dictionary of IOSurface's properties; the surface comes back retained.
        let made = NonNull::new(unsafe { IOSurfaceCreate(properties.0) }).map(Surface);
        made.ok_or_else(|| {
            format!(
                "IOSurfaceCreate answered nil for a {}x{} '{}' surface of {length} bytes",
                size[0],
                size[1],
                four_cc(ostype)
            )
        })
    }

    /// The surface, as Metal's `newTextureWithDescriptor:iosurface:plane:` takes it.
    pub fn as_ptr(&self) -> *mut IoSurfaceOpaque {
        self.0.as_ptr()
    }

    /// Whether the surface came out laid as `planes` asked, plane for plane.
    ///
    /// `IOSurfaceCreate` reads the plane dictionary as a request, so this is the one place that
    /// can say the request was granted. A surface laid differently is a failure and never a
    /// repitched copy: a case's recording names where the planes are.
    pub fn planes_are_as_asked(&self, planes: &[PlaneLayout; 2]) -> Result<(), String> {
        let surface = self.as_ptr();
        // SAFETY: plane queries of a live surface; the base addresses are the allocation's own and
        // are compared, never read through.
        let (count, base) = unsafe {
            (
                IOSurfaceGetPlaneCount(surface),
                IOSurfaceGetBaseAddress(surface).cast::<u8>(),
            )
        };
        if count != planes.len() {
            return Err(format!(
                "the surface has {count} planes, and the case states {}",
                planes.len()
            ));
        }
        for (index, asked) in planes.iter().enumerate() {
            // SAFETY: `index` is below the plane count read just above, of a live surface.
            let found = unsafe {
                PlaneLayout {
                    width: IOSurfaceGetWidthOfPlane(surface, index) as u64,
                    height: IOSurfaceGetHeightOfPlane(surface, index) as u64,
                    offset: IOSurfaceGetBaseAddressOfPlane(surface, index)
                        .cast::<u8>()
                        .offset_from(base) as u64,
                    bytes_per_row: IOSurfaceGetBytesPerRowOfPlane(surface, index) as u64,
                    bytes_per_element: IOSurfaceGetBytesPerElementOfPlane(surface, index) as u64,
                }
            };
            if &found != asked {
                return Err(format!(
                    "plane {index} came out {found:?}, and the case states {asked:?}"
                ));
            }
        }
        Ok(())
    }

    /// Write `bytes` from the surface's base, which is where a case's stand-in buffer begins.
    pub fn fill(&self, bytes: &[u8]) -> Result<(), String> {
        let surface = self.as_ptr();
        // SAFETY: the surface is live; its allocation is checked to hold `bytes`, and the lock
        // keeps the base address valid until the unlock.
        unsafe {
            let allocated = IOSurfaceGetAllocSize(surface);
            if allocated < bytes.len() {
                return Err(format!(
                    "the surface holds {allocated} bytes, and the case's buffer is {}",
                    bytes.len()
                ));
            }
            let locked = IOSurfaceLock(surface, 0, core::ptr::null_mut());
            if locked != 0 {
                return Err(format!("IOSurfaceLock answered {locked}"));
            }
            let base = IOSurfaceGetBaseAddress(surface).cast::<u8>();
            let written = if base.is_null() {
                Err("IOSurfaceGetBaseAddress answered null".to_owned())
            } else {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len());
                Ok(())
            };
            IOSurfaceUnlock(surface, 0, core::ptr::null_mut());
            written
        }
    }
}

/// `ostype` as its four characters.
pub fn four_cc(ostype: [u8; 4]) -> String {
    ostype
        .iter()
        .map(|&byte| {
            if byte.is_ascii_graphic() {
                byte as char
            } else {
                '?'
            }
        })
        .collect()
}

/// How many bytes of each plane hold texels, so a case's own layout can be checked against the
/// bytes it means to hand over: every element of every row of both planes must lie inside the
/// buffer, and a pitch narrower than a row of elements would put two rows' texels on top of each
/// other.
pub fn planes_hold_their_texels(planes: &[PlaneLayout; 2], length: usize) -> Result<(), String> {
    for (index, plane) in planes.iter().enumerate() {
        let row = plane.texel_bytes_per_row();
        if plane.bytes_per_row < row {
            return Err(format!(
                "plane {index}'s rows are {} bytes apart, and its {} elements of {} need {row}",
                plane.bytes_per_row, plane.width, plane.bytes_per_element
            ));
        }
        let end = plane.offset + plane.bytes_per_row * plane.height;
        if end > length as u64 {
            return Err(format!(
                "plane {index} ends at byte {end}, past the buffer's {length}"
            ));
        }
    }
    Ok(())
}

/// The planes CoreVideo lays out for a `size` surface of OSType `ostype`, or `None` where it makes
/// no buffer of that OSType and size.
///
/// The tests' guard on [`planes_of`] alone: it is Apple's own answer for the geometry, and the
/// derivation is held to it. Nothing the suite records calls it, so no recording depends on
/// CoreVideo being present.
#[cfg(test)]
pub fn core_video_planes(ostype: [u8; 4], size: [u32; 2]) -> Option<Vec<(u64, u64, u64)>> {
    let empty = dictionary(&[]);
    // SAFETY: CoreVideo's key constant, read once.
    let attributes = dictionary(&[(unsafe { kCVPixelBufferIOSurfacePropertiesKey }, empty.0)]);
    let mut buffer: CfType = core::ptr::null();
    // SAFETY: a buffer of the given size; `buffer` receives a retained object or stays null.
    let status = unsafe {
        CVPixelBufferCreate(
            core::ptr::null(),
            size[0] as usize,
            size[1] as usize,
            u32::from_be_bytes(ostype),
            attributes.0,
            &raw mut buffer,
        )
    };
    let buffer = Owned(buffer);
    if status != 0 || buffer.0.is_null() {
        return None;
    }
    // SAFETY: the buffer's surface, which the buffer keeps while it lives (borrowed).
    let surface = unsafe { CVPixelBufferGetIOSurface(buffer.0) };
    if surface.is_null() {
        return None;
    }
    // SAFETY: plane queries of a live surface, each plane below its count.
    unsafe {
        let count = IOSurfaceGetPlaneCount(surface);
        Some(
            (0..count)
                .map(|plane| {
                    (
                        IOSurfaceGetWidthOfPlane(surface, plane) as u64,
                        IOSurfaceGetHeightOfPlane(surface, plane) as u64,
                        IOSurfaceGetBytesPerElementOfPlane(surface, plane) as u64,
                    )
                })
                .collect(),
        )
    }
}

#[cfg(test)]
#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferIOSurfacePropertiesKey: CfType;
    fn CVPixelBufferCreate(
        allocator: CfAllocator,
        width: usize,
        height: usize,
        pixel_format: u32,
        attributes: CfType,
        out: *mut CfType,
    ) -> i32;
    fn CVPixelBufferGetIOSurface(buffer: CfType) -> *mut IoSurfaceOpaque;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two surfaces the suite's recorded cases lay a texture over: the OSType, the texture's
    /// extent, where the planes lie, the buffer's length, and the format the texture is of.
    ///
    /// They are written here rather than read out of the cases because this is the module under
    /// test: a case that changes its surface changes its own input fingerprint and says so, and
    /// this table is what says the geometry these shapes need is the geometry Apple lays out.
    const SURFACES: &[([u8; 4], [u32; 2], [(u64, u64); 2], usize, PixelFormat)] = &[
        // `cases::ycbcr_textures::TexelReads`.
        (
            *b"420v",
            [40, 24],
            [(0, 64), (64 * 24, 64)],
            4096,
            PixelFormat::YCBCR84202P,
        ),
        // `cases::ycbcr_minify`, both bands.
        (
            *b"x420",
            [3840, 2160],
            [(0, 7680), (2 * 3840 * 2160, 7680)],
            24_883_200,
            PixelFormat::YCBCR104202P,
        ),
    ];

    /// The geometry [`planes_of`] derives from a format is the geometry CoreVideo lays out for a
    /// buffer of the same OSType and size: each plane's size in texels and the bytes of its
    /// element.
    ///
    /// This is the guard that keeps the derivation honest. It is Apple's own answer, so a format
    /// whose planes are not shaped the way the bits and the subsampling say fails here instead of
    /// handing Metal a surface laid one way and a texture that reads it another.
    #[test]
    fn the_derivation_is_core_videos_own_layout() {
        for &(ostype, size, at, length, format) in SURFACES {
            let derived = planes_of(format, size, at).expect("the format's planes");
            planes_hold_their_texels(&derived, length).expect("the planes hold their texels");
            let Some(apple) = core_video_planes(ostype, size) else {
                panic!(
                    "CoreVideo made no {}x{} '{}' buffer",
                    size[0],
                    size[1],
                    four_cc(ostype)
                );
            };
            let ours: Vec<_> = derived
                .iter()
                .map(|plane| (plane.width, plane.height, plane.bytes_per_element))
                .collect();
            assert_eq!(
                ours,
                apple,
                "'{}' {}x{}: the derivation and CoreVideo's own layout",
                four_cc(ostype),
                size[0],
                size[1]
            );
        }
    }

    /// A surface made for each of those shapes comes out laid exactly as the case states, and the
    /// bytes a case hands over land where its planes say: the chroma plane's first byte is the
    /// buffer's byte at the chroma offset.
    ///
    /// Without this, a surface Apple's allocator laid its own way would be compared against bytes
    /// in other places, and the case would read as a divergence of ours.
    #[test]
    fn a_surface_comes_out_laid_as_the_case_states() {
        for &(ostype, size, at, length, format) in SURFACES {
            let planes = planes_of(format, size, at).expect("the format's planes");
            let surface = Surface::two_plane(ostype, size, &planes, length).expect("the surface");
            surface
                .planes_are_as_asked(&planes)
                .expect("the surface is laid as the case states");
            // A byte a plane's first row holds, distinct for each plane, so a copy that landed at
            // another offset is seen.
            let mut bytes = vec![0xa5u8; length];
            bytes[planes[0].offset as usize] = 0x11;
            bytes[planes[1].offset as usize] = 0x22;
            surface.fill(&bytes).expect("the fill");
            let raw = surface.as_ptr();
            // SAFETY: the planes' base addresses of a live surface, each inside the allocation the
            // surface just accepted `length` bytes into; one byte read from each.
            let (luma, chroma) = unsafe {
                (
                    *IOSurfaceGetBaseAddressOfPlane(raw, 0).cast::<u8>(),
                    *IOSurfaceGetBaseAddressOfPlane(raw, 1).cast::<u8>(),
                )
            };
            assert_eq!((luma, chroma), (0x11, 0x22), "'{}'", four_cc(ostype));
        }
    }

    /// A plane whose rows are closer together than its own elements need, or which ends past the
    /// buffer, is refused before a surface is made: both would put a texel somewhere the case did
    /// not say it was.
    #[test]
    fn a_plane_that_does_not_hold_its_texels_is_refused() {
        let planes = planes_of(PixelFormat::YCBCR84202P, [40, 24], [(0, 64), (1536, 64)])
            .expect("the format's planes");
        planes_hold_their_texels(&planes, 4096).expect("the case's own layout");
        // One byte short of the chroma plane's last row.
        let why = planes_hold_their_texels(&planes, 2303).expect_err("past the buffer");
        assert!(why.contains("plane 1 ends at byte 2304"), "{why}");
        // A luma pitch narrower than its 40 one-byte codes.
        let mut narrow = planes;
        narrow[0].bytes_per_row = 39;
        let why = planes_hold_their_texels(&narrow, 4096).expect_err("rows on top of each other");
        assert!(why.contains("plane 0's rows are 39 bytes apart"), "{why}");
    }

    /// A format that is not a two-plane YCbCr one, and a packed form this side does not lay out,
    /// are named rather than guessed at.
    #[test]
    fn a_format_without_two_planes_is_named() {
        let why = planes_of(PixelFormat::RGBA8Unorm, [8, 8], [(0, 32), (256, 32)])
            .expect_err("not a two-plane format");
        assert!(why.contains("is not a two-plane YCbCr format"), "{why}");
        let why = planes_of(
            PixelFormat::YCBCR104202PPACKED,
            [8, 8],
            [(0, 32), (256, 32)],
        )
        .expect_err("a packed form");
        assert!(why.contains("packs its codes"), "{why}");
    }
}
