//! The Metal side of the case vocabulary: native Metal through `objc2-metal`, on macOS only.
//!
//! This is what record mode runs. It exists so a case's Metal answer is taken by running the case
//! rather than copied off a terminal: the same case body, the same sites, the same transcript
//! shape.
//!
//! A shader reaches Metal as its fixture's typed-pointer AIR text: `xcrun metal-as` reassembles it
//! and `xcrun metallib` packages the library, which `newLibraryWithURL:` loads. Apple's assembler
//! reads that dialect and not the one a current `llvm-as` writes, which is why a fixture commits
//! both forms.
//!
//! Metal announces no degrade notice, so this side announces none: a notice in a transcript is
//! ours, and a case that holds one must name what justifies it.

use crate::argument_encoders::{ArgumentDescriptorSpec, ArgumentWrite, EncoderFacts};
use crate::case::{
    Answer, ArgumentSlot, Bindings, CommandDraw, Failure, FragmentFill, FragmentFilled, Handle,
    MeshCommand, MeshFunctions, MeshStep, PassOutcome, RateMapFacts, RateMapRequest, SamplerSpec,
    ShaderSource, Side, StageInRegion, StageInput, TextureRole, TextureSpec, TileFacts,
    TilePipelineSpec, TileStep, Venue,
};
use crate::compute_pass::{
    ComputePass, ComputePassOutcome, ComputeStep, Dispatching, PassResource,
};
use crate::function_bindings::{
    ArrayFacts, BindingFacts, BufferFacts, MemberFacts, PointerFacts, ReferenceFacts, StructFacts,
    TextureFacts, ThreadgroupFacts, TypeFacts,
};
use crate::function_facts::{
    AttributeFacts, ConstantAt, ConstantFacts, ConstantValue, FunctionFacts,
};
use crate::pipeline_facts::{ComputeStateFacts, ComputeStateSpec, RenderStateFacts};
use crate::queue_slots::QueueSlots;
use crate::render::{
    Dispatch, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, StageFunction, TargetSpec,
};
use crate::sparse_query::{
    HeapFacts, QueryPage, SparseHeapSpec, SparseTextureFacts, SparseTextureSpec, SparseTileQuery,
    TileSize,
};
use crate::texture_query::{HeapTextureQuery, SizeAndAlign, Storage};
use crate::transcript::Value;
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BindingAccess, BindingType, FunctionOptions, FunctionType, PatchType, PixelFormat, RenderStages,
};
use metal2vulkan_runtime::SparsePageSize;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSDictionary, NSObjectProtocol, NSRange, NSString, NSURL};
use objc2_metal::{
    MTLArgumentDescriptor, MTLArgumentEncoder, MTLArrayType, MTLAttribute, MTLAttributeFormat,
    MTLBarrierScope, MTLBinding, MTLBindingAccess, MTLBindingType, MTLBlendFactor,
    MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer, MTLBufferBinding, MTLClearColor,
    MTLColorWriteMask, MTLCommandBuffer, MTLCommandBufferHandler, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineReflection, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLDispatchType, MTLEvent, MTLFunction, MTLFunctionConstant,
    MTLFunctionConstantValues, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLIndexType,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectRenderCommand, MTLLibrary, MTLLoadAction, MTLMeshRenderPipelineDescriptor,
    MTLMultisampleDepthResolveFilter, MTLMultisampleStencilResolveFilter, MTLOrigin,
    MTLPipelineOption, MTLPixelFormat, MTLPointerType, MTLPrimitiveType,
    MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMap,
    MTLRasterizationRateMapDescriptor, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineReflection, MTLRenderPipelineState,
    MTLRenderStages, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSamplePosition,
    MTLSamplerAddressMode, MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLSharedEvent, MTLSize, MTLSparsePageSize, MTLStageInputOutputDescriptor,
    MTLStencilDescriptor, MTLStencilOperation, MTLStepFunction, MTLStorageMode, MTLStoreAction,
    MTLStoreActionOptions, MTLStructMember, MTLStructType, MTLTexture, MTLTextureBinding,
    MTLTextureDescriptor, MTLTextureReferenceType, MTLTextureSwizzle, MTLTextureSwizzleChannels,
    MTLTextureType, MTLTextureUsage, MTLThreadgroupBinding, MTLTileRenderPipelineDescriptor,
    MTLVertexAttribute, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction, MTLViewport,
    MTLVisibilityResultMode, MTLVisibleFunctionTable,
};

/// The Metal level this side answers at: the host's own, since Metal answers as the release it runs
/// on.
const METAL_LEVEL: &str = crate::recording::ORACLE_OS_VERSION;

/// Native Metal, with the objects the case has made.
pub struct MetalSide {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    libraries: Vec<Retained<ProtocolObject<dyn MTLLibrary>>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// The sampler states the case has made. Its own pool: a sampler state is its own Metal
    /// object, not another name for one of the pools above.
    samplers: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    command_buffers: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// The render pipeline states the case has made, each with the reflection its creation
    /// answered where the case asked for one. One pool, because a mesh state, a tile state and a
    /// graphics state are one Metal object (`MTLRenderPipelineState`) made from three descriptors.
    render_pipelines: Vec<RenderPipeline>,
    /// The indirect command buffers the case has made.
    command_buffers_indirect: Vec<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    /// The rasterization-rate maps the case has made. Its own pool: a rate map is its own
    /// Metal object, not another name for one of the pools above.
    rate_maps: Vec<Retained<ProtocolObject<dyn MTLRasterizationRateMap>>>,
    /// For each argument buffer the case has made, the textures it names: Metal keeps a resource a
    /// kernel reaches through an argument buffer resident only where the encoder was told so, so
    /// binding the buffer makes them resident. Not a pool — the buffer itself lives in `buffers`.
    argument_residents: Vec<(Handle, Vec<Handle>)>,
    /// The depth and stencil states the case has made. Its own pool: an `MTLDepthStencilState` is
    /// its own Metal object, not another name for one of the pools above.
    depth_stencil_states: Vec<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    /// The heaps the case has made. Its own pool: an `MTLHeap` is its own Metal object, not
    /// another name for one of the pools above, and not another name for the depth and stencil
    /// states either.
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    /// The argument encoders the case has made, each with the argument buffer it writes into. Its
    /// own pool: an `MTLArgumentEncoder` is its own Metal object, not another name for one of the
    /// pools above.
    argument_encoders: Vec<ArgumentEncoderAt>,
    /// **Every IOSurface this side made, and the only thing that owns one.** Not a pool a handle
    /// names: a surface is not a case's object at all — the case states a buffer, or a plane's
    /// rotation and stride, and this side turns it into the surface Metal needs. They are kept so
    /// a surface outlives the texture that reads it.
    ///
    /// The two families that need a surface — a two-plane YCbCr plane texture and a rotated
    /// framebuffer-only target — are **one mechanism here**, which is why there is one pool. What
    /// differs between them is the layout asked for and the geometry it comes from, and that lives
    /// in [`crate::metal_surface`]'s two constructors, not in a second owner.
    surfaces: Vec<crate::metal_surface::Surface>,
    /// Which surface a texture handle's plane reads and writes, and how long that plane is.
    ///
    /// **A lookup, not a pool**: it owns nothing and releases nothing, and the surface it names
    /// lives in `surfaces` above. It exists because a rotated target's case writes the plane
    /// before its pass and reads it after, so that side has to find the surface again from the
    /// texture handle — which a plane texture's case never does, since its bytes go in before the
    /// surface is handed to Metal and come back out through the texture.
    planes: Vec<PlaneAt>,
    /// The scratch directory the metallibs are built in, removed when the side is dropped.
    scratch: Scratch,
}

/// Where one texture handle's plane lies: which surface in [`MetalSide::surfaces`] holds it, and
/// how many bytes of that surface the plane is.
///
/// The index is into that vector, which only ever grows while a side lives, so an index stays the
/// surface it named. The length is the **case's** — its stride times its rows — and not
/// `IOSurfaceGetAllocSize`, which is routinely larger: measured here a 16x64 plane asked for a
/// 272-byte stride came back in 32,768 bytes, 15,360 more than the plane holds. Comparing the
/// allocation would compare bytes no case ever stated.
#[derive(Clone, Copy)]
struct PlaneAt {
    handle: Handle,
    at: usize,
    length: usize,
}

/// One render pipeline state of the case's, with the reflection its creation answered.
///
/// The reflection lives here, beside the state, rather than in a pool of its own, because **Metal
/// answers a reflection only at the state's creation**: `MTLRenderPipelineState`'s own
/// `reflection` property is nil for a state an `MTLDevice` made. So a handle has to stand for the
/// pair, and the state is kept so it outlives its own reflection. It is `None` for a state whose
/// creation was not asked for one, which is every state but
/// [`Side::render_pipeline_with_reflection`]'s.
struct RenderPipeline {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    reflection: Option<Retained<MTLRenderPipelineReflection>>,
}

impl RenderPipeline {
    /// A state whose creation was not asked for a reflection.
    fn unreflected(state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>) -> Self {
        Self {
            state,
            reflection: None,
        }
    }
}

/// One argument encoder of the case's, with where its arguments go.
///
/// The target is kept beside the encoder because `constantDataAtIndex:` answers an **address**,
/// and the fact a case compares is its offset inside the arguments: to take that offset the side
/// needs the argument buffer the encoder was given and where in it the arguments start.
struct ArgumentEncoderAt {
    encoder: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
    /// The argument buffer and the offset of the arguments in it; `None` until one is set.
    target: Option<(Handle, u64)>,
}

/// The array of Objective-C pointers one of `MTLArgumentEncoder`'s range setters takes, with a
/// null for each nil the case named, and the range it covers.
///
/// Its own function so the eight range setters share one spelling: each takes
/// `NonNull<*const ProtocolObject<dyn T>>` over the whole range, and a hand-written array per
/// setter is eight chances to get a length or a first index wrong.
fn pointer_array<T: ?Sized>(
    objects: &[Option<Retained<ProtocolObject<T>>>],
) -> Vec<*const ProtocolObject<T>> {
    objects
        .iter()
        .map(|object| match object {
            Some(object) => &**object as *const ProtocolObject<T>,
            None => std::ptr::null(),
        })
        .collect()
}

/// The range a range setter writes: `elements.len()` arguments from `first`.
///
/// Refused where the case named no element at all: Metal's array setters read `range.length`
/// pointers from the array they are handed, and an empty slice has no address to hand them. No
/// case states an empty range, and this says so rather than passing a dangling pointer.
fn range_of<T>(first: u32, elements: &[T]) -> Answer<NSRange> {
    if elements.is_empty() {
        return Err(Failure::named(
            "EmptyRange",
            "a range setter of no element".to_owned(),
        ));
    }
    Ok(NSRange::new(first as usize, elements.len()))
}

/// A binding as the subprotocol it conforms to, with the conformance **asked of the object**
/// rather than assumed from its `type`.
///
/// `objc2` offers no cast between two protocol objects: `ProtocolObject::from_ref` needs a static
/// conformance, which `dyn MTLBinding` to `dyn MTLBufferBinding` is not, and a protocol object
/// cannot be downcast. The alternative is `msg_send!`, which sends a selector with no receiver type
/// at all — and that is how `MTLTextureBinding`'s `arrayLength` came to be read by
/// `texture_accesses` while the coverage document still counted every one of its items unexercised:
/// a selector sent without a receiver is a call nothing, including that derivation, can check. So
/// the cast is spelled once, here, and `conformsToProtocol:` makes its precondition a test rather
/// than a comment.
fn conforming<'a, P: ?Sized>(
    binding: &'a ProtocolObject<dyn MTLBinding>,
    protocol: &std::ffi::CStr,
) -> Answer<&'a ProtocolObject<P>> {
    let declared = objc2::runtime::AnyProtocol::get(protocol)
        .ok_or_else(|| Failure::named("NoSuchProtocol", protocol.to_string_lossy().into_owned()))?;
    if !binding.conformsToProtocol(declared) {
        return Err(Failure::named(
            "NotConforming",
            format!(
                "a binding of MTLBindingType({}) does not conform to {}",
                binding.r#type().0,
                protocol.to_string_lossy()
            ),
        ));
    }
    // SAFETY: `ProtocolObject<P>` is a `repr(transparent)` wrapper over the same Objective-C
    // object whatever `P` is, and the object was just asked, and answered, that it conforms to
    // `P`'s protocol, so every selector `P` declares is one it responds to.
    Ok(unsafe { &*std::ptr::from_ref(binding).cast::<ProtocolObject<P>>() })
}

/// The translator's `MTLDataType` for a raw Metal ordinal.
fn our_data_type(raw: usize) -> Answer<metal2vulkan::metal::DataType> {
    metal2vulkan::metal::DataType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownDataType", format!("MTLDataType({raw})")))
}

/// The translator's `MTLFunctionType` for a raw Metal ordinal.
fn our_function_type(raw: usize) -> Answer<FunctionType> {
    FunctionType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownFunctionType", format!("MTLFunctionType({raw})")))
}

/// The translator's `MTLPatchType` for a raw Metal ordinal.
fn our_patch_type(raw: usize) -> Answer<PatchType> {
    PatchType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownPatchType", format!("MTLPatchType({raw})")))
}

/// The six facts an attribute states, read off either of the two classes that state them.
///
/// `MTLVertexAttribute` and `MTLAttribute` are two Objective-C classes with the same six getters
/// and no common protocol, so there is nothing to be generic over: the macro is what keeps the two
/// readings from drifting apart, which a second hand-written copy would not.
macro_rules! attribute_facts {
    ($list:expr) => {{
        let mut found = Vec::new();
        for attribute in $list {
            found.push(crate::function_facts::AttributeFacts {
                name: attribute.name().to_string(),
                attribute_index: attribute.attributeIndex() as u32,
                attribute_type: our_data_type(attribute.attributeType().0)?,
                active: attribute.isActive(),
                patch_data: attribute.isPatchData(),
                patch_control_point_data: attribute.isPatchControlPointData(),
            });
        }
        found
    }};
}

/// `vertexAttributes`' list in the vocabulary both sides answer in; `None` is Metal's nil.
fn our_vertex_attributes(
    list: Option<&NSArray<MTLVertexAttribute>>,
) -> Answer<Option<Vec<AttributeFacts>>> {
    match list {
        None => Ok(None),
        Some(list) => Ok(Some(attribute_facts!(list))),
    }
}

/// `stageInputAttributes`' list in the vocabulary both sides answer in; `None` is Metal's nil.
fn our_stage_input_attributes(
    list: Option<&NSArray<MTLAttribute>>,
) -> Answer<Option<Vec<AttributeFacts>>> {
    match list {
        None => Ok(None),
        Some(list) => Ok(Some(attribute_facts!(list))),
    }
}

/// The `MTLFunctionConstantValues` the case's values build, each through the setter its
/// [`ConstantAt`] names.
fn constant_values(values: &[ConstantValue]) -> Retained<MTLFunctionConstantValues> {
    let built = MTLFunctionConstantValues::new();
    for value in values {
        let bytes = std::ptr::NonNull::from(value.bytes.as_slice()).cast::<std::ffi::c_void>();
        let data_type = MTLDataType(value.data_type.as_raw() as usize);
        // SAFETY: each pointer is to `value.bytes`, which outlives the call, and each holds the
        // bytes the case states for the type and the count it states. The index, the range and
        // the name are the case's own; Metal checks them against the function's own constants
        // when the pipeline state compiles, and a value naming no constant is ignored.
        unsafe {
            match &value.at {
                ConstantAt::Index(index) => {
                    built.setConstantValue_type_atIndex(bytes, data_type, *index as usize);
                }
                ConstantAt::Range { first, count } => {
                    built.setConstantValues_type_withRange(
                        bytes,
                        data_type,
                        NSRange::new(*first as usize, *count as usize),
                    );
                }
                ConstantAt::Name(name) => {
                    built.setConstantValue_type_withName(
                        bytes,
                        data_type,
                        &NSString::from_str(name),
                    );
                }
            }
        }
    }
    built
}

/// The translator's `MTLBindingAccess` for a raw Metal ordinal.
fn our_access(raw: usize) -> Answer<BindingAccess> {
    BindingAccess::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownBindingAccess", format!("MTLBindingAccess({raw})")))
}

/// The translator's `MTLTextureType` for a raw Metal ordinal.
fn our_texture_type(raw: usize) -> Answer<metal2vulkan::metal::TextureType> {
    metal2vulkan::metal::TextureType::try_from(raw as u32)
        .map_err(|_| Failure::named("UnknownTextureType", format!("MTLTextureType({raw})")))
}

/// One binding of Metal's reflection, in the vocabulary both sides answer in.
///
/// Which subprotocol to read is decided by the binding's own `type`, which is how Metal states it:
/// `MTLBindingTypeBuffer` is the fact that the binding conforms to `MTLBufferBinding`.
fn metal_binding(binding: &ProtocolObject<dyn MTLBinding>) -> Answer<BindingFacts> {
    let kind = binding.r#type();
    let mut facts = BindingFacts {
        name: binding.name().to_string(),
        binding_type: BindingType::try_from(kind.0 as u32).map_err(|_| {
            Failure::named("UnknownBindingType", format!("MTLBindingType({})", kind.0))
        })?,
        access: our_access(binding.access().0)?,
        index: binding.index() as u32,
        used: binding.isUsed(),
        argument: binding.isArgument(),
        buffer: None,
        threadgroup: None,
        texture: None,
    };
    if kind == MTLBindingType::Buffer {
        let held: &ProtocolObject<dyn MTLBufferBinding> = conforming(binding, c"MTLBufferBinding")?;
        let struct_type = held.bufferStructType();
        let pointer_type = held.bufferPointerType();
        facts.buffer = Some(BufferFacts {
            alignment: held.bufferAlignment() as u64,
            data_size: held.bufferDataSize() as u64,
            data_type: our_data_type(held.bufferDataType().0)?,
            struct_type: struct_type
                .as_deref()
                .map(|held| metal_struct(held, crate::function_bindings::DEPTH))
                .transpose()?,
            pointer_type: pointer_type
                .as_deref()
                .map(|held| metal_pointer(held, crate::function_bindings::DEPTH))
                .transpose()?,
        });
    }
    if kind == MTLBindingType::ThreadgroupMemory {
        let held: &ProtocolObject<dyn MTLThreadgroupBinding> =
            conforming(binding, c"MTLThreadgroupBinding")?;
        facts.threadgroup = Some(ThreadgroupFacts {
            alignment: held.threadgroupMemoryAlignment() as u64,
            data_size: held.threadgroupMemoryDataSize() as u64,
        });
    }
    if kind == MTLBindingType::Texture {
        let held: &ProtocolObject<dyn MTLTextureBinding> =
            conforming(binding, c"MTLTextureBinding")?;
        facts.texture = Some(TextureFacts {
            texture_type: our_texture_type(held.textureType().0)?,
            data_type: our_data_type(held.textureDataType().0)?,
            depth: held.isDepthTexture(),
            array_length: held.arrayLength() as u32,
        });
    }
    Ok(facts)
}

/// A struct type of Metal's: its members, and whether `memberByName:` finds each of them.
fn metal_struct(held: &MTLStructType, depth: u32) -> Answer<StructFacts> {
    let members = held.members();
    let mut found = Vec::new();
    let mut by_name = true;
    for member in &members {
        let name = member.name();
        by_name &= held
            .memberByName(&name)
            .is_some_and(|other| other.name().to_string() == name.to_string());
        found.push(metal_member(&member, depth)?);
    }
    Ok(StructFacts {
        members: found,
        found_by_name: by_name,
    })
}

fn metal_member(member: &MTLStructMember, depth: u32) -> Answer<MemberFacts> {
    Ok(MemberFacts {
        name: member.name().to_string(),
        offset: member.offset() as u64,
        data_type: our_data_type(member.dataType().0)?,
        argument_index: member.argumentIndex() as u32,
        kind: match depth {
            0 => TypeFacts::Deeper,
            _ => metal_detail(
                member.structType().as_deref(),
                member.arrayType().as_deref(),
                member.textureReferenceType().as_deref(),
                member.pointerType().as_deref(),
                depth - 1,
            )?,
        },
    })
}

/// Which of the four type getters answered, in the order Metal declares them.
fn metal_detail(
    held_struct: Option<&MTLStructType>,
    held_array: Option<&MTLArrayType>,
    held_reference: Option<&MTLTextureReferenceType>,
    held_pointer: Option<&MTLPointerType>,
    depth: u32,
) -> Answer<TypeFacts> {
    if let Some(held) = held_struct {
        return Ok(TypeFacts::Struct(metal_struct(held, depth)?));
    }
    if let Some(held) = held_array {
        return Ok(TypeFacts::Array(Box::new(metal_array(held, depth)?)));
    }
    if let Some(held) = held_reference {
        return Ok(TypeFacts::TextureReference(metal_reference(held)?));
    }
    if let Some(held) = held_pointer {
        return Ok(TypeFacts::Pointer(Box::new(metal_pointer(held, depth)?)));
    }
    Ok(TypeFacts::Plain)
}

/// What a texture an argument buffer holds says it is.
///
/// A function of its own, like the three beside it, rather than four getters read where the type
/// getter answered: the receiver of a call is typed by what declares it, and a binding an `if let`
/// pattern made carries no declared type, so reading these four there named their container to
/// nobody. The same four calls, with the receiver a parameter, resolve to `MTLTextureReferenceType`.
fn metal_reference(held: &MTLTextureReferenceType) -> Answer<ReferenceFacts> {
    Ok(ReferenceFacts {
        data_type: our_data_type(held.textureDataType().0)?,
        texture_type: our_texture_type(held.textureType().0)?,
        access: our_access(held.access().0)?,
        depth: held.isDepthTexture(),
    })
}

fn metal_array(held: &MTLArrayType, depth: u32) -> Answer<ArrayFacts> {
    Ok(ArrayFacts {
        element_type: our_data_type(held.elementType().0)?,
        array_length: held.arrayLength() as u32,
        stride: held.stride() as u64,
        argument_index_stride: held.argumentIndexStride() as u32,
        element: match depth {
            0 => TypeFacts::Deeper,
            _ => metal_detail(
                held.elementStructType().as_deref(),
                held.elementArrayType().as_deref(),
                held.elementTextureReferenceType().as_deref(),
                held.elementPointerType().as_deref(),
                depth - 1,
            )?,
        },
    })
}

fn metal_pointer(held: &MTLPointerType, depth: u32) -> Answer<PointerFacts> {
    Ok(PointerFacts {
        element_type: our_data_type(held.elementType().0)?,
        access: our_access(held.access().0)?,
        alignment: held.alignment() as u64,
        data_size: held.dataSize() as u64,
        element_is_argument_buffer: held.elementIsArgumentBuffer(),
        element: match depth {
            0 => TypeFacts::Deeper,
            // A pointer declares no texture-reference getter of its own, which is the nil here.
            _ => metal_detail(
                held.elementStructType().as_deref(),
                held.elementArrayType().as_deref(),
                None,
                None,
                depth - 1,
            )?,
        },
    })
}

/// The head of a non-empty array, as the pointer a range setter takes.
fn array_head<T>(elements: &[T]) -> Answer<std::ptr::NonNull<T>> {
    elements
        .first()
        .map(std::ptr::NonNull::from)
        .ok_or_else(|| Failure::named("EmptyRange", "a range setter of no element".to_owned()))
}

impl MetalSide {
    /// The system default device and a queue on it.
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "MTLCreateSystemDefaultDevice returned nil".to_owned())?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "MTLDevice::newCommandQueue returned nil".to_owned())?;
        Ok(Self {
            device,
            queue,
            libraries: Vec::new(),
            pipelines: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            command_buffers: Vec::new(),
            render_pipelines: Vec::new(),
            command_buffers_indirect: Vec::new(),
            rate_maps: Vec::new(),
            argument_residents: Vec::new(),
            depth_stencil_states: Vec::new(),
            heaps: Vec::new(),
            argument_encoders: Vec::new(),
            surfaces: Vec::new(),
            planes: Vec::new(),
            scratch: Scratch::new()?,
        })
    }

    /// The buffer `handle` names.
    fn held(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLBuffer>> {
        self.buffers
            .get(handle.0 as usize)
            .map(|buffer| &**buffer)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("buffer {}", handle.0)))
    }

    /// The texture `handle` names.
    fn held_texture(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLTexture>> {
        self.textures
            .get(handle.0 as usize)
            .map(|texture| &**texture)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("texture {}", handle.0)))
    }

    /// The sampler state `handle` names.
    fn held_sampler(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLSamplerState>> {
        self.samplers
            .get(handle.0 as usize)
            .map(|sampler| &**sampler)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("sampler {}", handle.0)))
    }

    /// The argument encoder `handle` names.
    ///
    /// Retained rather than borrowed, because a case's write reaches the encoder and the pools of
    /// the resources it writes in one call, and the residency it then records needs the side
    /// mutably.
    fn held_encoder(
        &self,
        handle: Handle,
    ) -> Answer<Retained<ProtocolObject<dyn MTLArgumentEncoder>>> {
        self.argument_encoders
            .get(handle.0 as usize)
            .map(|at| at.encoder.clone())
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("argument encoder {}", handle.0)))
    }

    /// The heap `handle` names.
    fn held_heap(&self, handle: Handle) -> Answer<&ProtocolObject<dyn MTLHeap>> {
        self.heaps
            .get(handle.0 as usize)
            .map(|heap| &**heap)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("heap {}", handle.0)))
    }

    /// The rasterization-rate map `handle` names.
    fn held_rate_map(
        &self,
        handle: Handle,
    ) -> Answer<&ProtocolObject<dyn MTLRasterizationRateMap>> {
        self.rate_maps
            .get(handle.0 as usize)
            .map(|map| &**map)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("rate map {}", handle.0)))
    }

    /// Level 0 of `texture` read back through a blit into a shared buffer of its own, at
    /// `row_pitch` bytes a row: what a private texture's bytes are, since the host reaches none of
    /// them directly.
    fn blit_out(
        &mut self,
        texture: &ProtocolObject<dyn MTLTexture>,
        size: [u32; 2],
        row_pitch: u32,
    ) -> Answer<Vec<u8>> {
        let length = row_pitch as usize * size[1] as usize;
        let back = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let blit = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        // SAFETY: the buffer holds `size[1]` rows of `row_pitch` bytes, which is what the copy
        // writes, and the texture holds level 0 of that extent.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                texture,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize {
                    width: size[0] as usize,
                    height: size[1] as usize,
                    depth: 1,
                },
                &back,
                0,
                row_pitch as usize,
                length,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let mut bytes = vec![0u8; length];
        // SAFETY: the buffer is shared storage of at least `length` bytes and the copy into it has
        // completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                back.contents().as_ptr().cast::<u8>(),
                bytes.as_mut_ptr(),
                length,
            );
        }
        keep(&mut self.buffers, back);
        keep(&mut self.command_buffers, command_buffer);
        Ok(bytes)
    }

    /// One command buffer with one compute encoder: the pipeline set, `bindings` bound at 0 up,
    /// each of `stage_in` set in order, `dispatch` encoded, and `commit`. The two dispatch calls
    /// share this body, so neither side can pick up a different binding order with one of them.
    fn encode(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        dispatch: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Answer<Handle> {
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = bindings
            .buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let sampled: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = bindings
            .textures
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let states: Vec<Retained<ProtocolObject<dyn MTLSamplerState>>> = bindings
            .samplers
            .iter()
            .map(|handle| self.held_sampler(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        // The textures the bound argument buffers name: Metal keeps a resource reached through an
        // argument buffer resident only where the encoder was told so.
        let resident: Vec<Retained<ProtocolObject<dyn MTLResource>>> = self
            .argument_residents
            .iter()
            .filter(|(buffer, _)| bindings.buffers.contains(buffer))
            .flat_map(|(_, textures)| textures.clone())
            .map(|handle| {
                self.held_texture(handle).map(|texture| {
                    let resource: &ProtocolObject<dyn MTLResource> =
                        ProtocolObject::from_ref(texture);
                    Retained::from(resource)
                })
            })
            .collect::<Answer<Vec<_>>>()?;
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .clone();
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the kernel's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, at) };
        }
        for (at, texture) in sampled.iter().enumerate() {
            // SAFETY: the texture is of the kind and access the kernel declares at that slot, and
            // the encoder outlives the call.
            unsafe { encoder.setTexture_atIndex(Some(texture), at) };
        }
        for (at, sampler) in states.iter().enumerate() {
            // SAFETY: the kernel declares a sampler at that slot, and the encoder outlives the
            // call.
            unsafe { encoder.setSamplerState_atIndex(Some(sampler), at) };
        }
        for resource in &resident {
            encoder.useResource_usage(resource, MTLResourceUsage::Read);
        }
        for region in stage_in {
            match region {
                StageInRegion::Direct { origin, size } => {
                    encoder.setStageInRegion(MTLRegion {
                        origin: MTLOrigin {
                            x: origin[0] as usize,
                            y: origin[1] as usize,
                            z: origin[2] as usize,
                        },
                        size: MTLSize {
                            width: size[0] as usize,
                            height: size[1] as usize,
                            depth: size[2] as usize,
                        },
                    });
                }
                StageInRegion::Indirect { buffer, offset } => {
                    let held = self.held(*buffer)?;
                    // SAFETY: the buffer holds the region's six words from `offset`, which the
                    // case states as one of its inputs, and the encoder outlives the call.
                    unsafe {
                        encoder.setStageInRegionWithIndirectBuffer_indirectBufferOffset(
                            held,
                            *offset as usize,
                        );
                    }
                }
            }
        }
        dispatch(&encoder);
        encoder.endEncoding();
        command_buffer.commit();
        Ok(keep(&mut self.command_buffers, command_buffer))
    }
}

/// Three axes as Metal's size.
fn size(axes: [u32; 3]) -> MTLSize {
    MTLSize {
        width: axes[0] as usize,
        height: axes[1] as usize,
        depth: axes[2] as usize,
    }
}

/// `MTLBufferLayoutStrideDynamic`, the stride a layout takes from the encoder's binding.
const STRIDE_DYNAMIC: usize = usize::MAX;

/// Metal's vertex descriptor for `layout`. The case's enumerations are Metal's own, so each one
/// crosses as its raw value.
fn vertex_descriptor(layout: &crate::render::VertexLayout<'_>) -> Retained<MTLVertexDescriptor> {
    let descriptor = MTLVertexDescriptor::vertexDescriptor();
    for attribute in layout.attributes {
        // SAFETY: the index is below the attribute array every descriptor holds, which the case's
        // own guard keeps it to.
        let held = unsafe {
            descriptor
                .attributes()
                .objectAtIndexedSubscript(attribute.index as usize)
        };
        held.setFormat(MTLVertexFormat(attribute.format.as_raw() as usize));
        // SAFETY: the offset and the buffer index are the case's own numbers, both inside the
        // record the case binds and the buffer array the descriptor holds.
        unsafe {
            held.setOffset(attribute.offset as usize);
            held.setBufferIndex(attribute.buffer as usize);
        }
    }
    for buffer in layout.layouts {
        // SAFETY: as the attributes', for the layout array.
        let held = unsafe {
            descriptor
                .layouts()
                .objectAtIndexedSubscript(buffer.index as usize)
        };
        held.setStepFunction(MTLVertexStepFunction(buffer.step.as_raw() as usize));
        // SAFETY: the stride is the case's own record size and the rate is the step function's
        // own, 0 for the constant step and 1 otherwise.
        unsafe {
            held.setStride(buffer.stride as usize);
            held.setStepRate(buffer.rate as usize);
        }
    }
    descriptor
}

/// Metal's stage-input descriptor for `stage_input`. The case's enumerations are Metal's own, so
/// each reaches `MTLAttributeFormat`, `MTLStepFunction` and `MTLIndexType` by its raw ordinal.
fn stage_input_descriptor(stage_input: &StageInput) -> Retained<MTLStageInputOutputDescriptor> {
    let descriptor = MTLStageInputOutputDescriptor::stageInputOutputDescriptor();
    let attributes = descriptor.attributes();
    for attribute in &stage_input.attributes {
        // SAFETY: `objectAtIndexedSubscript:` is Metal's own accessor for an attribute index the
        // device's attribute limit covers, which the case's indices are inside.
        let written = unsafe { attributes.objectAtIndexedSubscript(attribute.index as usize) };
        written.setFormat(MTLAttributeFormat(attribute.format.as_raw() as usize));
        written.setOffset(attribute.offset as usize);
        // SAFETY: the buffer index is inside the device's buffer argument table, which the
        // case's indices are.
        unsafe { written.setBufferIndex(attribute.buffer_index as usize) };
    }
    let layouts = descriptor.layouts();
    for layout in &stage_input.layouts {
        // SAFETY: as above, for a buffer index inside the device's buffer limit.
        let written = unsafe { layouts.objectAtIndexedSubscript(layout.buffer_index as usize) };
        written.setStride(
            layout
                .stride
                .map_or(STRIDE_DYNAMIC, |stride| stride as usize),
        );
        written.setStepFunction(MTLStepFunction(layout.step_function.as_raw() as usize));
        written.setStepRate(layout.step_rate as usize);
    }
    descriptor.setIndexType(MTLIndexType(stage_input.index_type.as_raw() as usize));
    // SAFETY: as above, for the index buffer's own slot.
    unsafe { descriptor.setIndexBufferIndex(stage_input.index_buffer_index as usize) };
    descriptor
}

/// `kept` with `value` pushed on, answering its index.
/// The `MTLTextureSwizzle` a channel's source is, by the raw ordinal both enumerations share.
fn swizzle_of(source: metal2vulkan::metal::TextureSwizzle) -> MTLTextureSwizzle {
    MTLTextureSwizzle(source.as_raw() as u8)
}

fn keep<T>(kept: &mut Vec<T>, value: T) -> Handle {
    kept.push(value);
    Handle(u32::try_from(kept.len() - 1).expect("fewer than 2^32 objects"))
}

/// Metal's texture descriptor for the one a case states: every field of
/// [`TextureRequest`](crate::texture::TextureRequest) reaches it.
///
/// A function of its own, and not inlined in the call, so that
/// `both_sides_state_every_field_of_a_texture_request` can hold the two sides' descriptors to the
/// same request field by field with no device. Making an `MTLTextureDescriptor` asks nothing of a
/// device, so that guard runs wherever Metal's frameworks are.
///
/// [`crate::texture::Over`] is **not** here: it is which object the call is sent to, not a field of
/// the descriptor.
pub(crate) fn texture_descriptor(
    request: &crate::texture::TextureRequest<'_>,
) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setPixelFormat(MTLPixelFormat(request.format as usize));
    descriptor.setTextureType(MTLTextureType(request.texture_type.as_raw() as usize));
    // SAFETY: the setters take the descriptor's own words; nothing is borrowed.
    unsafe {
        descriptor.setWidth(request.size[0] as usize);
        descriptor.setHeight(request.size[1] as usize);
        descriptor.setDepth(request.size[2].max(1) as usize);
        descriptor.setMipmapLevelCount(request.levels.max(1) as usize);
        descriptor.setArrayLength(request.slices.max(1) as usize);
    }
    descriptor.setUsage(MTLTextureUsage(request.usage.as_raw() as usize));
    descriptor.setStorageMode(if request.private {
        MTLStorageMode::Private
    } else {
        MTLStorageMode::Shared
    });
    descriptor.setAllowGPUOptimizedContents(request.gpu_optimized);
    descriptor.setSwizzle(MTLTextureSwizzleChannels {
        red: swizzle_of(request.swizzle.red),
        green: swizzle_of(request.swizzle.green),
        blue: swizzle_of(request.swizzle.blue),
        alpha: swizzle_of(request.swizzle.alpha),
    });
    descriptor
}

/// A directory the metallibs are built in.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "metal2vulkan-conformance-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `xcrun tool args`, answering why it could not.
fn xcrun(tool: &str, args: &[&std::ffi::OsStr]) -> Result<(), String> {
    let output = std::process::Command::new("/usr/bin/xcrun")
        .arg(tool)
        .args(args)
        .output()
        .map_err(|error| format!("xcrun {tool}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "xcrun {tool} failed: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

impl MetalSide {
    /// The pipeline state each distinct kernel a program's per-sample read-backs name, in the
    /// order they are first named.
    ///
    /// Empty for a program with no such read-back, so a program of copies alone compiles nothing.
    fn read_back_pipelines(
        &mut self,
        work: &crate::work::Work,
    ) -> Answer<Vec<crate::metal_work::Kernel>> {
        let mut wanted: Vec<&'static str> = Vec::new();
        for step in &work.steps {
            if let crate::work::Step::ReadSamples { kernel, .. } = step {
                if !wanted.contains(kernel) {
                    wanted.push(kernel);
                }
            }
        }
        if wanted.is_empty() {
            return Ok(Vec::new());
        }
        let fixture = work.kernels.ok_or_else(|| {
            Failure::named("NoKernels", "the program states no fixture for its kernels")
        })?;
        let loaded = crate::fixture::Fixture::new(fixture)
            .loaded()
            .map_err(|why| Failure::named("NoFixture", why))?;
        let library = self.library(&loaded.source())?;
        let mut made = Vec::with_capacity(wanted.len());
        for kernel in wanted {
            let handle = self.compute_pipeline(library, kernel)?;
            let state = self
                .pipelines
                .get(handle.0 as usize)
                .cloned()
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", handle.0)))?;
            made.push((kernel, state));
        }
        Ok(made)
    }
}

impl Side for MetalSide {
    fn venue(&self) -> Venue {
        Venue {
            device: self.device.name().to_string(),
            driver: "Metal".to_owned(),
            api: "Metal".to_owned(),
            metal_level: METAL_LEVEL.to_owned(),
        }
    }

    fn library(&mut self, source: &ShaderSource<'_>) -> Answer<Handle> {
        let serial = self.libraries.len();
        let text = self.scratch.0.join(format!("{serial}.air.ll"));
        let air = self.scratch.0.join(format!("{serial}.air"));
        let metallib = self.scratch.0.join(format!("{serial}.metallib"));
        let build = || -> Result<(), String> {
            std::fs::write(&text, source.air_text)
                .map_err(|error| format!("{}: {error}", text.display()))?;
            xcrun(
                "metal-as",
                &[text.as_os_str(), "-o".as_ref(), air.as_os_str()],
            )?;
            xcrun(
                "metallib",
                &[air.as_os_str(), "-o".as_ref(), metallib.as_os_str()],
            )
        };
        build().map_err(|why| Failure::named("MetalToolFailed", why))?;
        let path = metallib
            .to_str()
            .ok_or_else(|| Failure::named("NonUtf8Path", metallib.display().to_string()))?;
        let library = self
            .device
            .newLibraryWithURL_error(&NSURL::fileURLWithPath(&NSString::from_str(path)))
            .map_err(|error| Failure::named("NewLibraryWithURL", error.to_string()))?;
        Ok(keep(&mut self.libraries, library))
    }

    fn compute_pipeline(&mut self, library: Handle, entry: &str) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn static_threadgroup_memory_length(&mut self, pipeline: Handle) -> Answer<u64> {
        let state = self
            .pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?;
        Ok(state.staticThreadgroupMemoryLength() as u64)
    }

    fn buffer(&mut self, bytes: &[u8]) -> Answer<Handle> {
        let handle = self.zeroed_buffer(bytes.len())?;
        let buffer = self.held(handle)?;
        // SAFETY: the buffer was just made with at least `bytes.len()` bytes of shared storage,
        // and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Ok(handle)
    }

    fn zeroed_buffer(&mut self, length: usize) -> Answer<Handle> {
        let buffer = self
            .device
            .newBufferWithLength_options(length.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{length} bytes returned nil"),
                )
            })?;
        Ok(keep(&mut self.buffers, buffer))
    }

    fn texture_2d(
        &mut self,
        format: PixelFormat,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(format.as_raw() as usize),
                size[0] as usize,
                size[1] as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{format:?} {}x{} returned nil", size[0], size[1]),
                )
            })?;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        if bytes.len() < row_pitch as usize * size[1] as usize {
            return Err(Failure::named(
                "TooFewBytes",
                format!("{} bytes for {} rows of {row_pitch}", bytes.len(), size[1]),
            ));
        }
        // SAFETY: `bytes` holds `size[1]` rows of `row_pitch` bytes, checked just above, which is
        // the region the call writes.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::from(bytes).cast(),
                row_pitch as usize,
            );
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn read_texture(&mut self, texture: Handle, size: [u32; 2], row_pitch: u32) -> Answer<Vec<u8>> {
        let texture = self.held_texture(texture)?;
        let mut bytes = vec![0u8; row_pitch as usize * size[1] as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` is `size[1]` rows of `row_pitch` bytes, which is the region the call
        // reads, and the caller waited for the command buffer that wrote the texture.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                row_pitch as usize,
                region,
                0,
            );
        }
        Ok(bytes)
    }

    fn texture_2d_in_role(&mut self, spec: &TextureSpec, bytes: &[u8]) -> Answer<Handle> {
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        descriptor.setUsage(match spec.role {
            TextureRole::Read => MTLTextureUsage::ShaderRead,
            TextureRole::Write => MTLTextureUsage::ShaderWrite,
            TextureRole::ReadWrite => MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
            TextureRole::RenderTarget => MTLTextureUsage::RenderTarget,
        });
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "{:?} {}x{} returned nil",
                        spec.format, spec.size[0], spec.size[1]
                    ),
                )
            })?;
        if !bytes.is_empty() {
            if bytes.len() < spec.row_pitch as usize * spec.size[1] as usize {
                return Err(Failure::named(
                    "TooFewBytes",
                    format!(
                        "{} bytes for {} rows of {}",
                        bytes.len(),
                        spec.size[1],
                        spec.row_pitch
                    ),
                ));
            }
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: spec.size[0] as usize,
                    height: spec.size[1] as usize,
                    depth: 1,
                },
            };
            // SAFETY: `bytes` holds the rows of the pitch the region writes, checked just above.
            unsafe {
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    std::ptr::NonNull::from(bytes).cast(),
                    spec.row_pitch as usize,
                );
            }
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn texture(&mut self, request: &crate::texture::TextureRequest<'_>) -> Answer<Handle> {
        let descriptor = texture_descriptor(request);
        let swizzle = descriptor.swizzle();
        // One of the three Metal calls that answers a texture of this descriptor, as the request
        // says: the device's own, a buffer's, or a parent texture's view.
        let made = match request.over {
            None => self.device.newTextureWithDescriptor(&descriptor),
            Some(crate::texture::Over::Buffer {
                buffer,
                offset,
                bytes_per_row,
            }) => {
                let held: Retained<ProtocolObject<dyn MTLBuffer>> =
                    Retained::from(self.held(buffer)?);
                held.newTextureWithDescriptor_offset_bytesPerRow(
                    &descriptor,
                    offset as usize,
                    bytes_per_row as usize,
                )
            }
            Some(crate::texture::Over::View { texture }) => {
                let held: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(texture)?);
                // SAFETY: the level and slice ranges lie inside the parent, which the case's own
                // request for the parent states, and the format and type are a view Metal makes of
                // it or nil.
                unsafe {
                    held.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                        MTLPixelFormat(request.format as usize),
                        MTLTextureType(request.texture_type.as_raw() as usize),
                        objc2_foundation::NSRange::new(0, request.levels.max(1) as usize),
                        objc2_foundation::NSRange::new(0, request.slices.max(1) as usize),
                        swizzle,
                    )
                }
            }
        };
        let texture = made.ok_or_else(|| {
            Failure::named(
                "NewTextureWithDescriptor",
                format!(
                    "format {} {:?} {}x{}x{} returned nil",
                    request.format,
                    request.texture_type,
                    request.size[0],
                    request.size[1],
                    request.size[2]
                ),
            )
        })?;
        for upload in request.uploads {
            if upload.bytes.len() < upload.needed() {
                return Err(Failure::named(
                    "TooFewBytes",
                    format!(
                        "{} bytes for {} rows of {}",
                        upload.bytes.len(),
                        upload.rows,
                        upload.row_pitch
                    ),
                ));
            }
            let region = MTLRegion {
                origin: MTLOrigin {
                    x: upload.origin[0] as usize,
                    y: upload.origin[1] as usize,
                    z: upload.origin[2] as usize,
                },
                size: MTLSize {
                    width: upload.size[0] as usize,
                    height: upload.size[1] as usize,
                    depth: upload.size[2].max(1) as usize,
                },
            };
            // SAFETY: `upload.bytes` holds the rows of the pitch the region writes, checked just
            // above, and the region lies in the level and slice the texture was made with.
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    upload.level as usize,
                    upload.slice as usize,
                    std::ptr::NonNull::from(upload.bytes).cast(),
                    upload.row_pitch as usize,
                    upload.image_pitch as usize,
                );
            }
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn replace_region(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        bytes: &[u8],
    ) -> Answer<()> {
        if bytes.len() < row_pitch as usize * size[1] as usize {
            return Err(Failure::named(
                "TooFewBytes",
                format!("{} bytes for {} rows of {row_pitch}", bytes.len(), size[1]),
            ));
        }
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::from(self.held_texture(texture)?);
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` holds `size[1]` rows of `row_pitch` bytes, checked just above, which is
        // the region the call writes.
        unsafe {
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                std::ptr::NonNull::from(bytes).cast(),
                row_pitch as usize,
            );
        }
        Ok(())
    }

    fn sampler(&mut self, spec: &SamplerSpec) -> Answer<Handle> {
        let filter = |filter: metal2vulkan::metal::SamplerMinMagFilter| {
            MTLSamplerMinMagFilter(filter.as_raw() as usize)
        };
        let address = |mode: metal2vulkan::metal::SamplerAddressMode| {
            MTLSamplerAddressMode(mode.as_raw() as usize)
        };
        let descriptor = MTLSamplerDescriptor::new();
        descriptor.setMinFilter(filter(spec.min_filter));
        descriptor.setMagFilter(filter(spec.mag_filter));
        descriptor.setSAddressMode(address(spec.s_address_mode));
        descriptor.setTAddressMode(address(spec.t_address_mode));
        descriptor.setBorderColor(MTLSamplerBorderColor(spec.border_color.as_raw() as usize));
        descriptor.setNormalizedCoordinates(spec.normalized_coordinates);
        descriptor.setSupportArgumentBuffers(spec.support_argument_buffers);
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewSamplerStateWithDescriptor",
                    format!("{spec:?} returned nil"),
                )
            })?;
        Ok(keep(&mut self.samplers, sampler))
    }

    fn argument_buffer(&mut self, slots: &[ArgumentSlot]) -> Answer<Handle> {
        let mut words = Vec::with_capacity(slots.len() * 8);
        let mut named = Vec::new();
        for slot in slots {
            let id = match *slot {
                ArgumentSlot::Texture(handle) => {
                    named.push(handle);
                    self.held_texture(handle)?.gpuResourceID().to_raw()
                }
                ArgumentSlot::Sampler(handle) => {
                    self.held_sampler(handle)?.gpuResourceID().to_raw()
                }
            };
            words.extend(id.to_le_bytes());
        }
        let buffer = self.buffer(&words)?;
        self.argument_residents.push((buffer, named));
        Ok(buffer)
    }

    fn argument_encoder(
        &mut self,
        library: Handle,
        entry: &str,
        buffer_index: u32,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        // SAFETY: the case names a buffer index the function really takes an argument buffer at,
        // which is what its own fixture declares; Metal raises for any other index, so a case of
        // that shape is not written here.
        let encoder = unsafe { function.newArgumentEncoderWithBufferIndex(buffer_index as usize) };
        Ok(keep(
            &mut self.argument_encoders,
            ArgumentEncoderAt {
                encoder,
                target: None,
            },
        ))
    }

    fn argument_encoder_with_arguments(
        &mut self,
        arguments: &[ArgumentDescriptorSpec],
    ) -> Answer<Handle> {
        let descriptors: Vec<Retained<MTLArgumentDescriptor>> = arguments
            .iter()
            .map(|spec| {
                let descriptor = MTLArgumentDescriptor::argumentDescriptor();
                descriptor.setDataType(MTLDataType(spec.data_type.as_raw() as usize));
                descriptor.setIndex(spec.index as usize);
                descriptor.setArrayLength(spec.array_length as usize);
                descriptor.setAccess(MTLBindingAccess(spec.access.as_raw() as usize));
                descriptor.setTextureType(MTLTextureType(spec.texture_type.as_raw() as usize));
                descriptor.setConstantBlockAlignment(spec.constant_block_alignment as usize);
                descriptor
            })
            .collect();
        let encoder = self
            .device
            .newArgumentEncoderWithArguments(&NSArray::from_retained_slice(&descriptors))
            .ok_or_else(|| {
                Failure::named(
                    "NewArgumentEncoderWithArguments",
                    format!("{} descriptor(s) returned nil", descriptors.len()),
                )
            })?;
        Ok(keep(
            &mut self.argument_encoders,
            ArgumentEncoderAt {
                encoder,
                target: None,
            },
        ))
    }

    fn argument_encoder_facts(&mut self, encoder: Handle, label: &str) -> Answer<EncoderFacts> {
        let registry = self.device.registryID();
        let held = self.held_encoder(encoder)?;
        held.setLabel(Some(&NSString::from_str(label)));
        Ok(EncoderFacts {
            encoded_length: held.encodedLength() as u64,
            alignment: held.alignment() as u64,
            label: held.label().map(|label| label.to_string()),
            // The registry ID is each host's own number, so only the **comparison** is portable:
            // the two sides agree that the encoder came from the case's device, never on the
            // number that says so.
            device_is_the_cases: held.device().registryID() == registry,
        })
    }

    fn set_argument_buffer(
        &mut self,
        encoder: Handle,
        buffer: Handle,
        offset: u64,
        element: Option<u64>,
    ) -> Answer<()> {
        let held_buffer: Retained<ProtocolObject<dyn MTLBuffer>> =
            Retained::from(self.held(buffer)?);
        let held = self.held_encoder(encoder)?;
        let arguments = offset + element.unwrap_or(0) * held.encodedLength() as u64;
        // SAFETY: the buffer outlives the encoder's use of it (this side's own pool owns it), and
        // the case states an offset whose arguments lie inside its own buffer.
        unsafe {
            match element {
                None => held.setArgumentBuffer_offset(Some(&held_buffer), offset as usize),
                Some(element) => held.setArgumentBuffer_startOffset_arrayElement(
                    Some(&held_buffer),
                    offset as usize,
                    element as usize,
                ),
            }
        }
        self.argument_encoders[encoder.0 as usize].target = Some((buffer, arguments));
        Ok(())
    }

    fn encode_argument(&mut self, encoder: Handle, write: &ArgumentWrite) -> Answer<()> {
        let target = self
            .argument_encoders
            .get(encoder.0 as usize)
            .and_then(|at| at.target)
            .map(|(buffer, _)| buffer);
        let textures = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| self.held_texture(h).map(Retained::from))
                        .transpose()
                })
                .collect()
        };
        let samplers = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| self.held_sampler(h).map(Retained::from))
                        .transpose()
                })
                .collect()
        };
        let buffers = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| handle.map(|h| self.held(h).map(Retained::from)).transpose())
                .collect()
        };
        let computes = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.pipelines
                                .get(h.0 as usize)
                                .map(|state| Retained::from(&**state))
                                .ok_or_else(|| {
                                    Failure::named("NoSuchHandle", format!("pipeline {}", h.0))
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        let renders = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.render_pipelines
                                .get(h.0 as usize)
                                .map(|held| Retained::from(&*held.state))
                                .ok_or_else(|| {
                                    Failure::named(
                                        "NoSuchHandle",
                                        format!("render pipeline {}", h.0),
                                    )
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        let commands = |handles: &[Option<Handle>]| -> Answer<Vec<_>> {
            handles
                .iter()
                .map(|handle| {
                    handle
                        .map(|h| {
                            self.command_buffers_indirect
                                .get(h.0 as usize)
                                .map(|held| Retained::from(&**held))
                                .ok_or_else(|| {
                                    Failure::named(
                                        "NoSuchHandle",
                                        format!("indirect command buffer {}", h.0),
                                    )
                                })
                        })
                        .transpose()
                })
                .collect()
        };
        // Every range setter takes a pointer to an array of Objective-C pointers over the whole
        // range, so each is written through this one shape.
        let held = self.held_encoder(encoder)?;
        match write {
            ArgumentWrite::Texture { texture, index } => {
                let held_textures = textures(&[*texture])?;
                // SAFETY: the texture outlives the call (this side's pool owns it), and the index
                // is the case's own, which Metal answers for or complains about itself.
                unsafe {
                    held.setTexture_atIndex(held_textures[0].as_deref(), *index as usize);
                }
            }
            ArgumentWrite::Textures {
                textures: ts,
                first,
            } => {
                let held_textures = textures(ts)?;
                let pointers = pointer_array(&held_textures);
                let range = range_of(*first, &pointers)?;
                // SAFETY: `pointers` is an array of `pointers.len()` Objective-C pointers, which
                // is the range's length, and each object it names outlives the call.
                unsafe {
                    held.setTextures_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::Sampler { sampler, index } => {
                let held_samplers = samplers(&[*sampler])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setSamplerState_atIndex(held_samplers[0].as_deref(), *index as usize);
                }
            }
            ArgumentWrite::Samplers {
                samplers: ss,
                first,
            } => {
                let held_samplers = samplers(ss)?;
                let pointers = pointer_array(&held_samplers);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setSamplerStates_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::Buffer {
                buffer,
                offset,
                index,
            } => {
                let held_buffers = buffers(&[*buffer])?;
                // SAFETY: as for `setTexture:atIndex:`; the offset is the case's own and lies
                // inside the buffer it names.
                unsafe {
                    held.setBuffer_offset_atIndex(
                        held_buffers[0].as_deref(),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            ArgumentWrite::Buffers { buffers: bs, first } => {
                let handles: Vec<Option<Handle>> = bs.iter().map(|(handle, _)| *handle).collect();
                let held_buffers = buffers(&handles)?;
                let pointers = pointer_array(&held_buffers);
                let offsets: Vec<usize> = bs.iter().map(|(_, at)| *at as usize).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: both arrays hold the range's length of elements, and each buffer they
                // name outlives the call.
                unsafe {
                    held.setBuffers_offsets_withRange(
                        array_head(&pointers)?,
                        std::ptr::NonNull::from(&offsets[0]),
                        range,
                    );
                }
            }
            ArgumentWrite::ComputePipelineState { state, index } => {
                let held_states = computes(&[*state])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setComputePipelineState_atIndex(
                        held_states[0].as_deref(),
                        *index as usize,
                    );
                }
            }
            ArgumentWrite::ComputePipelineStates { states, first } => {
                let held_states = computes(states)?;
                let pointers = pointer_array(&held_states);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setComputePipelineStates_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::RenderPipelineState { state, index } => {
                let held_states = renders(&[*state])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setRenderPipelineState_atIndex(held_states[0].as_deref(), *index as usize);
                }
            }
            ArgumentWrite::RenderPipelineStates { states, first } => {
                let held_states = renders(states)?;
                let pointers = pointer_array(&held_states);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setRenderPipelineStates_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::IndirectCommandBuffer { commands: c, index } => {
                let held_commands = commands(&[*c])?;
                // SAFETY: as for `setTexture:atIndex:`.
                unsafe {
                    held.setIndirectCommandBuffer_atIndex(
                        held_commands[0].as_deref(),
                        *index as usize,
                    );
                }
            }
            ArgumentWrite::IndirectCommandBuffers { commands: c, first } => {
                let held_commands = commands(c)?;
                let pointers = pointer_array(&held_commands);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe {
                    held.setIndirectCommandBuffers_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::VisibleFunctionTable { table, index } => match table {
                // SAFETY: nil is what the case states, and the index is its own.
                None => unsafe { held.setVisibleFunctionTable_atIndex(None, *index as usize) },
                // No case states one, and neither side holds a pool of them, so both answer the
                // same "no such call" rather than one of them inventing an object.
                Some(_) => return Err(Failure::unavailable("visible_function_table_handle")),
            },
            ArgumentWrite::VisibleFunctionTables { tables, first } => {
                if tables.iter().any(Option::is_some) {
                    return Err(Failure::unavailable("visible_function_table_handle"));
                }
                let pointers: Vec<*const ProtocolObject<dyn MTLVisibleFunctionTable>> =
                    tables.iter().map(|_| std::ptr::null()).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`, over an array of nils.
                unsafe {
                    held.setVisibleFunctionTables_withRange(array_head(&pointers)?, range);
                }
            }
            ArgumentWrite::Constant { index, bytes } => {
                // SAFETY: the index is the case's own; the call answers where the constant goes,
                // and Metal complains itself where no argument has it.
                let at = unsafe { held.constantDataAtIndex(*index as usize) };
                // SAFETY: the address is the constant's inside the argument buffer's contents, and
                // the case states a constant no longer than the argument it names, which its own
                // layout facts compare.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        at.as_ptr().cast::<u8>(),
                        bytes.len(),
                    );
                }
            }
        }
        // A texture a function reaches through an argument buffer is resident only where the
        // encoder was told so, so binding the buffer makes it resident.
        if let Some(buffer) = target {
            let named = write.textures();
            if !named.is_empty() {
                self.argument_residents.push((buffer, named));
            }
        }
        Ok(())
    }

    fn compute_pipeline_bindings(
        &mut self,
        library: Handle,
        entry: &str,
    ) -> Answer<Vec<BindingFacts>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let mut reflection = None;
        // SAFETY: the function is of this library, and `reflection` is this call's own
        // out-parameter.
        let state = unsafe {
            self.device
                .newComputePipelineStateWithFunction_options_reflection_error(
                    &function,
                    MTLPipelineOption::BindingInfo,
                    Some(&mut reflection),
                )
        }
        .map_err(|error| {
            Failure::named(
                "NewComputePipelineState",
                error.localizedDescription().to_string(),
            )
        })?;
        // Declared, not inferred: the coverage derivation types a receiver from what bound it,
        // and an `Option` the creation filled in through an out-parameter names no type at all.
        let reflection: Retained<MTLComputePipelineReflection> = reflection.ok_or_else(|| {
            Failure::named(
                "NoReflection",
                "the creation answered no reflection".to_owned(),
            )
        })?;
        // Bound to a local rather than called in the `for` header: the coverage derivation reads
        // a receiver by walking back from the dot, and in a `for` header that walk swallows the
        // loop variable, so the call would resolve to no type at all.
        let bindings = reflection.bindings();
        let mut found = Vec::new();
        for binding in &bindings {
            found.push(metal_binding(&binding)?);
        }
        // The state is kept so it outlives the reflection its creation answered, as the render
        // pipelines' pool does.
        keep(&mut self.pipelines, state);
        Ok(found)
    }

    fn compute_pipeline_with_descriptor(
        &mut self,
        library: Handle,
        spec: &ComputeStateSpec<'_>,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(spec.entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", spec.entry.to_owned()))?;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        // Set only where the case states one. `setLabel:nil` on a pipeline descriptor fails an
        // assertion inside Metal and ends the process ("label must not be nil", measured on this
        // M3 Max while recording this family), and Metal's own default is already nil, so a case
        // that means a nil label is a case that makes this call not at all.
        if let Some(label) = spec.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        descriptor.setSupportIndirectCommandBuffers(spec.support_indirect_command_buffers);
        if let Some(threads) = spec.required_threads_per_threadgroup {
            descriptor.setRequiredThreadsPerThreadgroup(size(threads));
        }
        let state = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| {
                Failure::named(
                    "NewComputePipelineState",
                    error.localizedDescription().to_string(),
                )
            })?;
        Ok(keep(&mut self.pipelines, state))
    }

    fn compute_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<ComputeStateFacts> {
        let registry = self.device.registryID();
        // Declared, so the coverage derivation types the receiver: the pool's own element type is
        // this side's `Vec`, which names no Metal type the walk can read off a field.
        let state: &Retained<ProtocolObject<dyn MTLComputePipelineState>> = self
            .pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", state.0)))?;
        let required = state.requiredThreadsPerThreadgroup();
        Ok(ComputeStateFacts {
            label: state.label().map(|label| label.to_string()),
            // The registry ID is each host's own number, so only the **comparison** is portable.
            device_is_the_cases: state.device().registryID() == registry,
            static_threadgroup_memory_length: state.staticThreadgroupMemoryLength() as u64,
            support_indirect_command_buffers: state.supportIndirectCommandBuffers(),
            // SAFETY: the dimensions are the case's own three positive extents, which the call
            // multiplies by the kernel's imageblock cell size and nothing else.
            imageblock_memory_length: unsafe {
                state.imageblockMemoryLengthForDimensions(size(imageblock))
            } as u64,
            required_threads_per_threadgroup: [
                required.width as u64,
                required.height as u64,
                required.depth as u64,
            ],
        })
    }

    fn render_pipeline_facts(
        &mut self,
        state: Handle,
        imageblock: [u32; 3],
    ) -> Answer<RenderStateFacts> {
        let registry = self.device.registryID();
        let state: &Retained<ProtocolObject<dyn MTLRenderPipelineState>> = &self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", state.0)))?
            .state;
        let required = state.requiredThreadsPerTileThreadgroup();
        Ok(RenderStateFacts {
            label: state.label().map(|label| label.to_string()),
            device_is_the_cases: state.device().registryID() == registry,
            max_total_threads_per_threadgroup: state.maxTotalThreadsPerThreadgroup() as u64,
            threadgroup_size_matches_tile_size: state.threadgroupSizeMatchesTileSize(),
            imageblock_sample_length: Some(state.imageblockSampleLength() as u64),
            imageblock_memory_length: Some(
                state.imageblockMemoryLengthForDimensions(size(imageblock)) as u64,
            ),
            support_indirect_command_buffers: state.supportIndirectCommandBuffers(),
            required_threads_per_tile_threadgroup: [
                required.width as u64,
                required.height as u64,
                required.depth as u64,
            ],
        })
    }

    fn function_facts(
        &mut self,
        library: Handle,
        entry: &str,
        constants: &[ConstantValue],
    ) -> Answer<FunctionFacts> {
        let registry = self.device.registryID();
        let library: &Retained<ProtocolObject<dyn MTLLibrary>> = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let name = NSString::from_str(entry);
        let function: Retained<ProtocolObject<dyn MTLFunction>> = match constants.is_empty() {
            true => library
                .newFunctionWithName(&name)
                .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?,
            false => {
                let values = constant_values(constants);
                library
                    .newFunctionWithName_constantValues_error(&name, &values)
                    .map_err(|error| {
                        Failure::named(
                            "NewFunctionWithConstantValues",
                            error.localizedDescription().to_string(),
                        )
                    })?
            }
        };
        // Each list bound to a local of its declared type, as the bindings walk does: the
        // coverage derivation types a receiver from what bound it, so a call made inside a `for`
        // header or on an `Option`'s contents names no type at all.
        let vertex: Option<Retained<NSArray<MTLVertexAttribute>>> = function.vertexAttributes();
        let stage_in: Option<Retained<NSArray<MTLAttribute>>> = function.stageInputAttributes();
        let dictionary: Retained<NSDictionary<NSString, MTLFunctionConstant>> =
            function.functionConstantsDictionary();
        let mut found = Vec::new();
        for key in dictionary.allKeys() {
            let constant: Retained<MTLFunctionConstant> = dictionary
                .objectForKey(&key)
                .ok_or_else(|| Failure::named("NoSuchConstant", key.to_string()))?;
            found.push(ConstantFacts {
                name: constant.name().to_string(),
                data_type: our_data_type(constant.r#type().0)?,
                index: constant.index() as u32,
                required: constant.required(),
            });
        }
        // A dictionary has no order, so the facts are sorted here and the comparison is of two
        // sorted lists: `allKeys` is free to answer in any order and a case still compares the
        // same constant against the same constant.
        found.sort_by(|left, right| left.name.cmp(&right.name));
        let count = function.patchControlPointCount();
        Ok(FunctionFacts {
            name: function.name().to_string(),
            function_type: our_function_type(function.functionType().0)?,
            patch_type: our_patch_type(function.patchType().0)?,
            // Metal answers -1 where there is no count, which the vocabulary spells `None`.
            patch_control_point_count: u32::try_from(count).ok(),
            vertex_attributes: our_vertex_attributes(vertex.as_deref())?,
            stage_input_attributes: our_stage_input_attributes(stage_in.as_deref())?,
            constants: found,
            options: FunctionOptions::try_from(function.options().0 as u32)
                .map_err(|error| Failure::named("UnknownFunctionOptions", error.to_string()))?,
            // The registry ID is each host's own number, so only the **comparison** is portable:
            // the two sides agree that the function came from the case's device, never on the
            // number that says so.
            device_is_the_cases: function.device().registryID() == registry,
        })
    }

    fn constant_data_offset(&mut self, encoder: Handle, index: u32) -> Answer<u64> {
        let (buffer, arguments) = self
            .argument_encoders
            .get(encoder.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("argument encoder {}", encoder.0))
            })?
            .target
            .ok_or_else(|| {
                Failure::named("NoArgumentBuffer", "no argument buffer is set".to_owned())
            })?;
        let base = self.held(buffer)?.contents().as_ptr() as usize + arguments as usize;
        let held = self.held_encoder(encoder)?;
        // SAFETY: the index is the case's own; the call answers an address inside the argument
        // buffer's contents, and nothing is read or written through it here.
        let at = unsafe { held.constantDataAtIndex(index as usize) };
        (at.as_ptr() as usize)
            .checked_sub(base)
            .map(|offset| offset as u64)
            .ok_or_else(|| {
                Failure::named(
                    "ConstantBeforeTheArguments",
                    "constantDataAtIndex: answered an address below the arguments".to_owned(),
                )
            })
    }

    fn dispatch(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        threadgroups: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, &[], |encoder| {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(threadgroups), size(threads));
        })
    }

    fn compute_pipeline_with_stage_input(
        &mut self,
        library: Handle,
        entry: &str,
        stage_input: &StageInput,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", entry.to_owned()))?;
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setStageInputDescriptor(Some(&stage_input_descriptor(stage_input)));
        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewComputePipelineState", error.to_string()))?;
        Ok(keep(&mut self.pipelines, pipeline))
    }

    fn dispatch_threads(
        &mut self,
        pipeline: Handle,
        bindings: &Bindings<'_>,
        stage_in: &[StageInRegion],
        grid: [u32; 3],
        threads: [u32; 3],
    ) -> Answer<Handle> {
        self.encode(pipeline, bindings, stage_in, |encoder| {
            encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(threads));
        })
    }

    fn wait(&mut self, command_buffer: Handle) -> Answer<Value> {
        let command_buffer = self
            .command_buffers
            .get(command_buffer.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("command buffer {}", command_buffer.0),
                )
            })?;
        command_buffer.waitUntilCompleted();
        Ok(Value::unit_variant(status_word(command_buffer.status())))
    }

    fn read_buffer(&mut self, buffer: Handle, offset: u64, length: usize) -> Answer<Vec<u8>> {
        let buffer = self.held(buffer)?;
        let past = offset
            .checked_add(length as u64)
            .is_none_or(|end| end > buffer.length() as u64);
        if past {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{length} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        // SAFETY: `[offset, offset + length)` is inside the buffer's shared storage, and the
        // caller waited for the command buffer that wrote it.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                buffer.contents().as_ptr().cast::<u8>().add(offset as usize),
                length,
            )
        };
        Ok(bytes.to_vec())
    }

    /// A copy-and-fill program, through public Metal alone ([`crate::metal_work`]).
    ///
    /// The colour fill (`fillTextureColor:`) is Apple SPI with no arm here, so a case built on one
    /// still carries its Metal answer. Everything else a program states -- heaps, placed textures,
    /// textures over a buffer, views, every blit, `replaceRegion:`, `getBytes:`, a clearing render
    /// pass and a per-sample read-back -- is public Metal whatever pixel format the case names,
    /// including Apple's two-plane extended-range 550 and 551; the pattern fill is reached by
    /// selector after `respondsToSelector:`.
    fn work(&mut self, work: &crate::work::Work) -> Answer<crate::work::Done> {
        // A per-sample read-back's pipeline states are made here and handed to the interpreter,
        // because making one needs the scratch directory, `xcrun metal-as`, `xcrun metallib` and
        // the library pool this side owns, none of which the interpreter sees.
        let kernels = self.read_back_pipelines(work)?;
        crate::metal_work::run(&self.device, &self.queue, work, &kernels)
    }

    fn announced(&mut self) -> Vec<String> {
        Vec::new()
    }

    fn mesh_pipeline_for_commands(
        &mut self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        descriptor.setSupportIndirectCommandBuffers(true);
        self.made_mesh_pipeline(&descriptor)
    }

    fn mesh_command_buffer(&mut self, commands: &[MeshCommand]) -> Answer<Handle> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(
            MTLIndirectCommandType::DrawMeshThreadgroups | MTLIndirectCommandType::DrawMeshThreads,
        );
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        // SAFETY: one object and one mesh binding, which is what every command of a case binds.
        unsafe {
            descriptor.setMaxObjectBufferBindCount(1);
            descriptor.setMaxMeshBufferBindCount(1);
        }
        // SAFETY: the count is the commands the case encodes, and every command below is inside
        // it.
        let made = unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &descriptor,
                    commands.len().max(1),
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or_else(|| {
            Failure::named(
                "NewIndirectCommandBuffer",
                format!("{} commands returned nil", commands.len()),
            )
        })?;
        for (at, command) in commands.iter().enumerate() {
            // SAFETY: `at` is below the buffer's command count, just set to `commands.len()`.
            let encoded = unsafe { made.indirectRenderCommandAtIndex(at) };
            let state = &self
                .render_pipelines
                .get(command.pipeline.0 as usize)
                .ok_or_else(|| {
                    Failure::named(
                        "NoSuchHandle",
                        format!("mesh pipeline {}", command.pipeline.0),
                    )
                })?
                .state;
            encoded.setRenderPipelineState(state);
            for (buffer, index) in &command.object_buffers {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // command outlives the call.
                unsafe { encoded.setObjectBuffer_offset_atIndex(buffer, 0, *index as usize) };
            }
            for (buffer, index) in &command.mesh_buffers {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe { encoded.setMeshBuffer_offset_atIndex(buffer, 0, *index as usize) };
            }
            let size = |axes: [u32; 3]| MTLSize {
                width: axes[0] as usize,
                height: axes[1] as usize,
                depth: axes[2] as usize,
            };
            match command.draw {
                CommandDraw::Threadgroups {
                    groups,
                    object,
                    mesh,
                } => encoded
                    .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        size(groups),
                        size(object),
                        size(mesh),
                    ),
                CommandDraw::Threads {
                    threads,
                    object,
                    mesh,
                } => encoded.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(threads),
                    size(object),
                    size(mesh),
                ),
            }
        }
        Ok(keep(&mut self.command_buffers_indirect, made))
    }

    fn execute_mesh_commands(
        &mut self,
        targets: &[Handle],
        commands: Handle,
        used: &[Handle],
    ) -> Answer<PassOutcome> {
        let icb = self
            .command_buffers_indirect
            .get(commands.0 as usize)
            .ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("command buffer {}", commands.0))
            })?
            .clone();
        let resident: Vec<&ProtocolObject<dyn MTLResource>> = used
            .iter()
            .map(|handle| self.held(*handle).map(ProtocolObject::from_ref))
            .collect::<Answer<Vec<_>>>()?;
        let (command_buffer, encoder) = self.pass(targets)?;
        let stages = MTLRenderStages::Object | MTLRenderStages::Mesh | MTLRenderStages::Fragment;
        for resource in &resident {
            encoder.useResource_usage_stages(
                resource,
                MTLResourceUsage::Read | MTLResourceUsage::Write,
                stages,
            );
        }
        // SAFETY: the range is the buffer's own command count.
        unsafe {
            encoder.executeCommandsInBuffer_withRange(
                &icb,
                objc2_foundation::NSRange::new(0, icb.size()),
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        // Metal's calls answer nothing, so no step of this side can refuse: one `None` per
        // resident resource, one for the execution and one for the commit, as the runtime side
        // answers them.
        let refused = vec![None; resident.len() + 2];
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn fragment_fill(&mut self, fill: &FragmentFill<'_>) -> Answer<FragmentFilled> {
        let format = MTLPixelFormat(fill.format.as_raw() as usize);
        let (vertex, fragment) = {
            let library = self.libraries.get(fill.library.0 as usize).ok_or_else(|| {
                Failure::named("NoSuchHandle", format!("library {}", fill.library.0))
            })?;
            let function = |name: &str| {
                library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
            };
            (function(fill.vertex)?, function(fill.fragment)?)
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        // SAFETY: attachment 0 is inside the eight every Metal pipeline descriptor holds.
        let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(format);
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| Failure::named("NewRenderPipelineState", error.to_string()))?;
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let target_descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                fill.size[0] as usize,
                fill.size[1] as usize,
                false,
            )
        };
        target_descriptor.setUsage(MTLTextureUsage::RenderTarget);
        target_descriptor.setStorageMode(MTLStorageMode::Private);
        let target = self
            .device
            .newTextureWithDescriptor(&target_descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "{:?} {}x{} render target returned nil",
                        fill.format, fill.size[0], fill.size[1]
                    ),
                )
            })?;
        let bound: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = fill
            .buffers
            .iter()
            .map(|handle| self.held(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: as the pipeline's attachment 0.
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(&target));
        color.setLoadAction(MTLLoadAction::DontCare);
        color.setStoreAction(MTLStoreAction::Store);
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        encoder.setRenderPipelineState(&state);
        for (index, buffer) in bound.iter().enumerate() {
            // SAFETY: the buffer is at least as long as the function's binding, and the encoder
            // outlives the call.
            unsafe { encoder.setFragmentBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        // SAFETY: the pipeline is set, the vertex function reads no buffer, and the count is the
        // case's own small one.
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                fill.vertices as usize,
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        let bytes = self.blit_out(&target, fill.size, fill.row_pitch)?;
        keep(&mut self.textures, target);
        Ok(FragmentFilled { status, bytes })
    }

    fn mesh_pipeline(&mut self, library: Handle, functions: &MeshFunctions<'_>) -> Answer<Handle> {
        let descriptor = self.mesh_descriptor(library, functions)?;
        self.made_mesh_pipeline(&descriptor)
    }

    fn render_target(&mut self, width: u32, height: u32) -> Answer<Handle> {
        // SAFETY: the extent is a case's own small one, and the descriptor is this call's own.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                TARGET_PIXEL_FORMAT,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{width}x{height} returned nil"),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn mesh_pass(&mut self, targets: &[Handle], steps: &[MeshStep]) -> Answer<PassOutcome> {
        let (command_buffer, encoder) = self.pass(targets)?;
        // Metal's encoder methods answer nothing, so no step of this side can refuse. A refusal
        // of ours then compares against Metal's answer and reads as a refusal.
        let refused = vec![None; steps.len()];
        for step in steps {
            self.encode_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn rate_map(&mut self, request: &RateMapRequest<'_>) -> Answer<Handle> {
        let descriptor = unsafe {
            MTLRasterizationRateMapDescriptor::rasterizationRateMapDescriptorWithScreenSize(
                MTLSize {
                    width: request.screen[0] as usize,
                    height: request.screen[1] as usize,
                    depth: 0,
                },
            )
        };
        if let Some(label) = request.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        for (at, layer) in request.layers.iter().enumerate() {
            // `initWithSampleCount:horizontal:vertical:` reads `width` samples from `horizontal`
            // and `height` from `vertical`, and answers nil for an axis of none, which is why
            // `RateMapLayer` holds at least one sample on each.
            let count = MTLSize {
                width: layer.horizontal.len(),
                height: layer.vertical.len(),
                depth: 0,
            };
            // SAFETY: each pointer is to at least as many floats as its axis's count, and Metal
            // copies the samples during the call.
            let made = unsafe {
                MTLRasterizationRateLayerDescriptor::initWithSampleCount_horizontal_vertical(
                    MTLRasterizationRateLayerDescriptor::alloc(),
                    count,
                    std::ptr::NonNull::new(layer.horizontal.as_ptr().cast_mut())
                        .ok_or_else(|| Failure::named("NoSamples", "a layer axis of none"))?,
                    std::ptr::NonNull::new(layer.vertical.as_ptr().cast_mut())
                        .ok_or_else(|| Failure::named("NoSamples", "a layer axis of none"))?,
                )
            };
            // SAFETY: the index is inside the array the descriptor grows on demand.
            unsafe { descriptor.setLayer_atIndex(Some(&made), at) };
        }
        let map = self
            .device
            .newRasterizationRateMapWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("NewRasterizationRateMap", "returned nil".to_owned()))?;
        Ok(keep(&mut self.rate_maps, map))
    }

    fn rate_map_facts(&mut self, map: Handle, layers: &[u64]) -> Answer<RateMapFacts> {
        let map = self.held_rate_map(map)?;
        let size = |size: MTLSize| [size.width as u64, size.height as u64, size.depth as u64];
        let sized = map.parameterBufferSizeAndAlign();
        Ok(RateMapFacts {
            label: map.label().map(|label| label.to_string()),
            screen_size: size(map.screenSize()),
            physical_granularity: size(map.physicalGranularity()),
            layer_count: map.layerCount() as u64,
            // SAFETY: `physicalSizeForLayer:` is not bounds-checked, and the case asks only about
            // layer indices Metal's own answers are measured at.
            physical_sizes: layers
                .iter()
                .map(|layer| size(unsafe { map.physicalSizeForLayer(*layer as usize) }))
                .collect(),
            parameter_size: sized.size as u64,
            parameter_align: sized.align as u64,
        })
    }

    fn copy_rate_map_parameters(&mut self, map: Handle, into: Handle, offset: u64) -> Answer<()> {
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from(self.held(into)?);
        let map = self.held_rate_map(map)?;
        if u64::try_from(buffer.length()).unwrap_or(u64::MAX)
            < offset + map.parameterBufferSizeAndAlign().size as u64
        {
            // Metal writes past the buffer's length here, so the case never asks it to: a short
            // buffer is refused before the copy rather than measured.
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{} parameter bytes at {offset} of a buffer of {}",
                    map.parameterBufferSizeAndAlign().size,
                    buffer.length()
                ),
            ));
        }
        // SAFETY: the buffer is of shared storage, is long enough for the parameter data at the
        // offset, is kept alive by `self.buffers`, and no command uses it.
        unsafe { map.copyParameterDataToBuffer_offset(&buffer, offset as usize) };
        Ok(())
    }

    fn read_texture_over(
        &mut self,
        texture: Handle,
        size: [u32; 2],
        row_pitch: u32,
        before: &[u8],
    ) -> Answer<Vec<u8>> {
        let wanted = row_pitch as usize * size[1] as usize;
        if before.len() != wanted {
            return Err(Failure::named(
                "TooFewBytes",
                format!(
                    "{} bytes before the read, and it reads {wanted}",
                    before.len()
                ),
            ));
        }
        let texture = self.held_texture(texture)?;
        let mut bytes = before.to_vec();
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: size[0] as usize,
                height: size[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `bytes` is `size[1]` rows of `row_pitch` bytes, checked just above, which is the
        // region the call reads, and no command buffer of this case is in flight.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::from(bytes.as_mut_slice()).cast(),
                row_pitch as usize,
                region,
                0,
            );
        }
        Ok(bytes)
    }

    fn render_pipeline_with_reflection(
        &mut self,
        library: Handle,
        vertex: &str,
        fragment: &str,
        attachment: PixelFormat,
    ) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| -> Answer<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NewFunctionWithName", name.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        let (vertex, fragment) = (function(vertex)?, function(fragment)?);
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        let attachments = descriptor.colorAttachments();
        // SAFETY: attachment 0 is inside the fixed-size array every descriptor holds.
        let first = unsafe { attachments.objectAtIndexedSubscript(0) };
        first.setPixelFormat(MTLPixelFormat(attachment.as_raw() as usize));
        let mut reflection = None;
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::BindingInfo,
                Some(&mut reflection),
            )
            .map_err(|error| {
                Failure::named(
                    "NewRenderPipelineState",
                    error.localizedDescription().to_string(),
                )
            })?;
        let reflection = reflection.ok_or_else(|| {
            Failure::named(
                "NoReflection",
                "the creation answered no reflection".to_owned(),
            )
        })?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline {
                state,
                reflection: Some(reflection),
            },
        ))
    }

    fn texture_accesses(
        &mut self,
        pipeline: Handle,
        stage: RenderStages,
        indices: &[u64],
    ) -> Answer<Vec<Option<BindingAccess>>> {
        let reflection = self
            .render_pipelines
            .get(pipeline.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0)))?
            .reflection
            .as_ref()
            .ok_or_else(|| {
                // Not a Metal gap and not a venue's: the case asked a state whose creation never
                // asked Metal for a reflection, and Metal cannot be asked afterwards.
                Failure::named(
                    "NoReflection",
                    format!("pipeline {} was not created with one", pipeline.0),
                )
            })?;
        // One stage, because that is how Metal states it: a binding list a stage. A union has no
        // Metal answer, only a composition rule of ours.
        let bindings = match stage.as_raw() {
            raw if raw == RenderStages::VERTEX.as_raw() => reflection.vertexBindings(),
            raw if raw == RenderStages::FRAGMENT.as_raw() => reflection.fragmentBindings(),
            raw if raw == RenderStages::TILE.as_raw() => reflection.tileBindings(),
            raw if raw == RenderStages::OBJECT.as_raw() => reflection.objectBindings(),
            raw if raw == RenderStages::MESH.as_raw() => reflection.meshBindings(),
            raw => {
                return Err(Failure::unavailable(&format!(
                    "texture_accesses of MTLRenderStages({raw:#x}), which Metal's reflection \
                     states no binding list for"
                )))
            }
        };
        // Each texture binding covers `arrayLength` slots from its own index, and an argument the
        // function never references is listed with `isUsed` false.
        let mut covered: Vec<(u64, u64, BindingAccess, bool)> = Vec::new();
        for binding in &bindings {
            if binding.r#type() != MTLBindingType::Texture {
                continue;
            }
            let access = match binding.access() {
                MTLBindingAccess::ReadOnly => BindingAccess::ReadOnly,
                MTLBindingAccess::ReadWrite => BindingAccess::ReadWrite,
                MTLBindingAccess::WriteOnly => BindingAccess::WriteOnly,
                other => {
                    return Err(Failure::named(
                        "UnknownBindingAccess",
                        format!("MTLBindingAccess({})", other.0),
                    ))
                }
            };
            // SAFETY: `MTLTextureBinding` extends `MTLBinding` with `arrayLength`, and the
            // binding answered `MTLBindingTypeTexture` just above, so it conforms to it. objc2
            // offers no safe cast between two protocol objects, so the selector is sent directly.
            let length: usize = unsafe { objc2::msg_send![&*binding, arrayLength] };
            covered.push((
                binding.index() as u64,
                length as u64,
                access,
                binding.isUsed(),
            ));
        }
        Ok(indices
            .iter()
            .map(|index| {
                covered
                    .iter()
                    .find(|(first, length, _, _)| *index >= *first && *index < *first + *length)
                    .and_then(|(_, _, access, used)| used.then_some(*access))
            })
            .collect())
    }

    fn tile_pipeline(&mut self, library: Handle, spec: &TilePipelineSpec<'_>) -> Answer<Handle> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(spec.entry))
            .ok_or_else(|| Failure::named("NoSuchFunction", spec.entry.to_owned()))?;
        let descriptor = MTLTileRenderPipelineDescriptor::new();
        // SAFETY: the function is of this library, and the descriptor is this call's own.
        unsafe { descriptor.setTileFunction(&function) };
        for (at, format) in spec.attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments the descriptor's array holds.
            unsafe {
                descriptor
                    .colorAttachments()
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(MTLPixelFormat(format.as_raw() as usize));
            }
        }
        if let Some(limit) = spec.max_total_threads_per_threadgroup {
            descriptor.setMaxTotalThreadsPerThreadgroup(limit as usize);
        }
        descriptor.setThreadgroupSizeMatchesTileSize(spec.threadgroup_size_matches_tile_size);
        let state = self
            .device
            .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewTileRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    fn tile_facts(&mut self, state: Handle) -> Answer<TileFacts> {
        let state = &self
            .render_pipelines
            .get(state.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0)))?
            .state;
        Ok(TileFacts {
            max_total_threads_per_threadgroup: state.maxTotalThreadsPerThreadgroup() as u64,
            threadgroup_size_matches_tile_size: state.threadgroupSizeMatchesTileSize(),
            imageblock_sample_length: Some(state.imageblockSampleLength() as u64),
        })
    }

    fn tile_pass(
        &mut self,
        targets: &[Handle],
        tile: [u32; 2],
        steps: &[TileStep],
    ) -> Answer<PassOutcome> {
        let (command_buffer, encoder) = self.loading_pass(targets, tile)?;
        // Metal's encoder methods answer nothing, so no step of this side can refuse.
        let refused = vec![None; steps.len()];
        for step in steps {
            self.encode_tile_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn compute_pass(&mut self, pass: &ComputePass<'_>) -> Answer<ComputePassOutcome> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        // The type is written out rather than inferred from the `match`, so that the coverage
        // derivation can resolve the receiver of the getter below: a call whose receiver does not
        // type is attributed to no item at all, which is how `dispatchType` came to read
        // unexercised while this very line read it.
        let encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> =
            match pass.dispatching {
                Dispatching::Serial => command_buffer.computeCommandEncoder(),
                Dispatching::Concurrent => command_buffer
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent),
            }
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", pass.dispatching.call()))?;
        // Read into its own binding, rather than as the scrutinee of the `match`: the coverage
        // derivation reads a call's receiver as the expression before the dot, and `match encoder`
        // is not a receiver it can type — which is how this very line left `dispatchType` reading
        // unexercised.
        let answered = encoder.dispatchType();
        let dispatch_type = Value::unit_variant(match answered {
            MTLDispatchType::Concurrent => "Concurrent",
            _ => "Serial",
        });
        // Metal's encoder methods answer nothing, so no step of this side can refuse. A refusal of
        // ours then compares against Metal's answer at that step and reads as a refusal.
        let refused = vec![None; pass.steps.len()];
        for step in pass.steps {
            self.encode_compute_step(&encoder, step)?;
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(ComputePassOutcome {
            refused,
            dispatch_type,
            status,
            error,
        })
    }

    fn target(&mut self, spec: &TargetSpec) -> Answer<Handle> {
        if let Some(plane) = &spec.surface {
            return self.surface_target(spec, plane);
        }
        if let Some(over) = &spec.over_buffer {
            return self.buffer_backed_target(spec, over);
        }
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        descriptor.setStorageMode(match spec.shared {
            true => MTLStorageMode::Shared,
            false => MTLStorageMode::Private,
        });
        if spec.samples > 1 {
            descriptor.setTextureType(MTLTextureType::Type2DMultisample);
            // SAFETY: the count is 2 or 4, which the case's own guard keeps it to, and Metal takes
            // both on every device; the descriptor is this call's own.
            unsafe { descriptor.setSampleCount(spec.samples as usize) };
        }
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!(
                        "a {:?} {}x{} target at {} samples returned nil",
                        spec.format, spec.size[0], spec.size[1], spec.samples
                    ),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    fn render_pipeline(
        &mut self,
        library: Handle,
        spec: &RenderPipelineSpec<'_>,
    ) -> Answer<Handle> {
        let function = |stage: &StageFunction<'_>| {
            let named = stage.library.unwrap_or(library);
            let held = self
                .libraries
                .get(named.0 as usize)
                .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", named.0)))?;
            held.newFunctionWithName(&NSString::from_str(stage.entry))
                .ok_or_else(|| Failure::named("NoSuchFunction", stage.entry.to_owned()))
        };
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&*function(&spec.vertex)?));
        if let Some(fragment) = &spec.fragment {
            descriptor.setFragmentFunction(Some(&*function(fragment)?));
        }
        // Set only where the case states one: `setLabel:nil` ends Metal's process, as the
        // compute descriptor's does.
        if let Some(label) = spec.label {
            descriptor.setLabel(Some(&NSString::from_str(label)));
        }
        descriptor.setSupportIndirectCommandBuffers(spec.support_indirect_command_buffers);
        descriptor.setRasterSampleCount(spec.samples as usize);
        descriptor.setRasterizationEnabled(spec.rasterization);
        if let Some(layout) = &spec.vertex_layout {
            descriptor.setVertexDescriptor(Some(&vertex_descriptor(layout)));
        }
        if let Some(format) = spec.depth_format {
            descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat(format.as_raw() as usize));
        }
        if let Some(format) = spec.stencil_format {
            descriptor.setStencilAttachmentPixelFormat(MTLPixelFormat(format.as_raw() as usize));
        }
        for (at, color) in spec.colors.iter().enumerate() {
            let Some(color) = color else { continue };
            // SAFETY: `at` is below the eight color attachments every Metal pipeline holds, which
            // the case's own guard keeps it to.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setPixelFormat(MTLPixelFormat(color.format.as_raw() as usize));
            attachment.setWriteMask(MTLColorWriteMask(color.write_mask.as_raw() as usize));
            if let Some(blend) = &color.blend {
                let factor = |factor: metal2vulkan::metal::BlendFactor| {
                    MTLBlendFactor(factor.as_raw() as usize)
                };
                let operation = |operation: metal2vulkan::metal::BlendOperation| {
                    MTLBlendOperation(operation.as_raw() as usize)
                };
                attachment.setBlendingEnabled(true);
                attachment.setSourceRGBBlendFactor(factor(blend.source_rgb));
                attachment.setDestinationRGBBlendFactor(factor(blend.destination_rgb));
                attachment.setRgbBlendOperation(operation(blend.rgb));
                attachment.setSourceAlphaBlendFactor(factor(blend.source_alpha));
                attachment.setDestinationAlphaBlendFactor(factor(blend.destination_alpha));
                attachment.setAlphaBlendOperation(operation(blend.alpha));
            }
        }
        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|error| Failure::named("NewRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    fn depth_stencil_state(&mut self, spec: &DepthStencilDescriptor) -> Answer<Handle> {
        let descriptor = MTLDepthStencilDescriptor::new();
        descriptor.setDepthCompareFunction(compare_function(spec.depth_compare_function));
        descriptor.setDepthWriteEnabled(spec.depth_write_enabled);
        descriptor.setFrontFaceStencil(Some(&stencil_descriptor(&spec.front_face_stencil)));
        descriptor.setBackFaceStencil(Some(&stencil_descriptor(&spec.back_face_stencil)));
        let state = self
            .device
            .newDepthStencilStateWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("NewDepthStencilState", "returned nil".to_owned()))?;
        Ok(keep(&mut self.depth_stencil_states, state))
    }

    fn write_plane(&mut self, target: Handle, bytes: &[u8]) -> Answer<()> {
        let (surface, length) = self.held_plane(target)?;
        if bytes.len() != length {
            return Err(Failure::named(
                "PlaneLength",
                format!("{} bytes for a plane of {length}", bytes.len()),
            ));
        }
        // No command buffer is in flight: a case writes the plane before it opens its pass and
        // reads it after the wait.
        surface
            .fill(bytes)
            .map_err(|why| Failure::named("WritePlane", why))
    }

    fn read_plane(&mut self, target: Handle) -> Answer<Vec<u8>> {
        let (surface, length) = self.held_plane(target)?;
        // The caller waited for the command buffer that wrote the plane before asking for it.
        surface
            .read(length)
            .map_err(|why| Failure::named("ReadPlane", why))
    }

    fn render(&mut self, work: &[RenderWork]) -> Answer<PassOutcome> {
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        // Metal's encoder methods answer nothing, so no piece of this side's program can refuse. A
        // refusal of ours then compares against Metal's answer and reads as a refusal.
        let refused = vec![None; work.len()];
        for piece in work {
            match piece {
                RenderWork::Pass(pass) => self.encode_pass(&command_buffer, pass)?,
                RenderWork::ReadTexture {
                    texture,
                    into,
                    offset,
                    row_pitch,
                    size,
                } => self.encode_read_texture(
                    &command_buffer,
                    *texture,
                    *into,
                    *offset,
                    *row_pitch,
                    *size,
                )?,
                RenderWork::Dispatch(dispatch) => {
                    self.encode_dispatch(&command_buffer, dispatch)?;
                }
            }
        }
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let error = command_buffer
            .error()
            .map(|error| Value::unit_variant(format!("code {}", error.code())));
        let status = Value::unit_variant(status_word(command_buffer.status()));
        keep(&mut self.command_buffers, command_buffer);
        Ok(PassOutcome {
            refused,
            status,
            error,
        })
    }

    fn heap_texture_size_and_align(&mut self, query: &HeapTextureQuery) -> Answer<SizeAndAlign> {
        let descriptor = queried_descriptor(query);
        let answered = self
            .device
            .heapTextureSizeAndAlignWithDescriptor(&descriptor);
        Ok(SizeAndAlign {
            size: answered.size as u64,
            align: answered.align as u64,
        })
    }

    fn queue_slots(&mut self, max: Option<u32>, cap: u32) -> Answer<QueueSlots> {
        let queue = match max {
            None => self.device.newCommandQueue(),
            Some(max) => self
                .device
                .newCommandQueueWithMaxCommandBufferCount(max as usize),
        }
        .ok_or_else(|| Failure::named("NewCommandQueue", format!("{max:?} returned nil")))?;
        let other = self
            .device
            .newCommandQueue()
            .ok_or_else(|| Failure::named("NewCommandQueue", "the second queue returned nil"))?;
        crate::queue_slots::probe(
            cap,
            move || {
                // Each make gets an autorelease pool of its own, and that is load-bearing rather
                // than tidy. `commandBuffer` answers an autoreleased object; the `Retained` is a
                // second reference, so without a pool that drains, the probe's own release frees
                // nothing, the slot never comes back, and the blocked make it is meant to release
                // stays blocked (measured: the whole probe timed out). A worker thread has no pool
                // of its own, so one is made here.
                objc2::rc::autoreleasepool(|_| {
                    queue
                        .commandBuffer()
                        .map(Made)
                        .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))
                })
            },
            || objc2::rc::autoreleasepool(|_| other.commandBuffer().is_some()),
        )
    }

    fn shared_event_rises(&mut self, label: &str) -> Answer<crate::shared_events::EventRise> {
        use crate::shared_events::{EventRise, WAIT_MS};
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil".to_owned()))?;
        event.setLabel(Some(&NSString::from_str(label)));
        let read_label = event.label().map(|label| label.to_string());
        let initial = event.signaledValue();
        // The host's sets: a lower value leaves the event where it was.
        event.setSignaledValue(5);
        event.setSignaledValue(3);
        event.setSignaledValue(5);
        let after_host_sets = event.signaledValue();
        let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
        let signal = |value: u64| -> Answer<u64> {
            // One autorelease pool per signal: the command buffer this answers is autoreleased,
            // and a loop that makes several without a pool that drains keeps every one of them
            // alive.
            objc2::rc::autoreleasepool(|_| {
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
                command_buffer.encodeSignalEvent_value(as_event, value);
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                Ok(event.signaledValue())
            })
        };
        let after_gpu_signals = [signal(3)?, signal(5)?];
        let waits = [
            event.waitUntilSignaledValue_timeoutMS(3, WAIT_MS),
            event.waitUntilSignaledValue_timeoutMS(5, WAIT_MS),
            event.waitUntilSignaledValue_timeoutMS(6, WAIT_MS),
        ];
        let after_max_signal = signal(u64::MAX)?;
        let waited_for_max = event.waitUntilSignaledValue_timeoutMS(u64::MAX, WAIT_MS);
        Ok(EventRise {
            label: read_label,
            initial,
            after_host_sets,
            after_gpu_signals,
            waits,
            after_max_signal,
            waited_for_max,
        })
    }

    fn event_cross_queue_wait(&mut self) -> Answer<crate::event_orders::CrossQueueWait> {
        use crate::event_orders::{
            CrossQueueWait, Status, HALF, SCRATCH, SIGNALLING_BYTE, WAITING_BYTE,
        };
        // One pool around the whole probe: `commandBuffer`, `newEvent` and `newCommandQueue` each
        // answer an autoreleased object, and a probe's thread has no pool of its own. Without this
        // the command buffers would live in the never-drained top-level pool and hold their
        // queues' slots for good.
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let (waiting_queue, signalling_queue) = (queue()?, queue()?);
            let event = self
                .device
                .newEvent()
                .ok_or_else(|| Failure::named("NewEvent", "returned nil"))?;
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let fill = |command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
                        offset: u64,
                        length: u64,
                        byte: u8|
             -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(offset as usize, length as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let waiting = waiting_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            waiting.encodeWaitForEvent_value(&event, 1);
            fill(&waiting, 0, HALF, WAITING_BYTE)?;
            let signalling = signalling_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&signalling, HALF, HALF, SIGNALLING_BYTE)?;
            signalling.encodeSignalEvent_value(&event, 1);
            // Bounded completion, rather than `waitUntilCompleted`, which Metal has no bounded
            // form of: the handler is added before the commit and the probe waits on its channel.
            let waiting_done = completion_channel(&waiting);
            let signalling_done = completion_channel(&signalling);
            // The waiting one first, which is the whole point: its signal is not committed yet.
            waiting.commit();
            let waiting_after_its_commit = Status::read(status_word(waiting.status()));
            signalling.commit();
            let signalling_at_the_end = settled(&signalling_done, &signalling);
            let waiting_at_the_end = settled(&waiting_done, &waiting);
            // SAFETY: `SCRATCH` bytes of the shared buffer, whose writers have both completed.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    scratch.contents().as_ptr().cast::<u8>(),
                    SCRATCH as usize,
                )
            }
            .to_vec();
            Ok(CrossQueueWait {
                waiting_after_its_commit,
                waiting_at_the_end,
                signalling_at_the_end,
                bytes,
            })
        })
    }

    fn event_highest_signal(&mut self, values: [u64; 2]) -> Answer<u64> {
        use crate::event_orders::SCRATCH;
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let queues = [queue()?, queue()?];
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))?;
            let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let mut done = Vec::new();
            for (queue, &value) in queues.iter().zip(&values) {
                let command_buffer = queue
                    .commandBuffer()
                    .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    value as u8,
                );
                encoder.endEncoding();
                command_buffer.encodeSignalEvent_value(as_event, value);
                let channel = completion_channel(&command_buffer);
                command_buffer.commit();
                done.push(channel);
            }
            for channel in &done {
                if channel
                    .done
                    .recv_timeout(crate::event_orders::COMPLETION)
                    .is_err()
                {
                    return Err(Failure::named(
                        "NeverCompleted",
                        "a signalling command buffer",
                    ));
                }
            }
            Ok(event.signaledValue())
        })
    }

    fn held_command_buffer(&mut self) -> Answer<crate::event_orders::HeldCommandBuffer> {
        use crate::event_orders::{
            HeldCommandBuffer, Status, COMPLETED, COMPLETION, HELD_FOR, SCHEDULED, SCRATCH,
        };
        objc2::rc::autoreleasepool(|_| {
            let queue = || {
                self.device
                    .newCommandQueue()
                    .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))
            };
            let (held_queue, other_queue) = (queue()?, queue()?);
            let event = self
                .device
                .newSharedEvent()
                .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))?;
            let as_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let fill = |command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
                        byte: u8|
             -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let mut log = HandlerLog::new();
            let held = held_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            held.encodeWaitForEvent_value(as_event, 1);
            fill(&held, 1)?;
            // SAFETY: each block is a valid heap block the log keeps alive for as long as this
            // probe runs, which outlives every call Metal makes of it.
            log.add(SCHEDULED, |block| unsafe {
                held.addScheduledHandler(block)
            });
            log.add(COMPLETED, |block| unsafe {
                held.addCompletedHandler(block)
            });
            held.commit();
            let behind = held_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&behind, 2)?;
            let behind_done = completion_channel(&behind);
            behind.commit();
            let other = other_queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            fill(&other, 3)?;
            let other_done = completion_channel(&other);
            other.commit();
            let other_queue_status = settled(&other_done, &other);
            // The window the negative facts are false in: a submission that was going to happen
            // has had its chance by now.
            std::thread::sleep(HELD_FOR);
            let held_while_waiting = Status::read(status_word(held.status()));
            let behind_while_waiting = Status::read(status_word(behind.status()));
            let handlers_while_waiting = log.names();
            event.setSignaledValue(1);
            // The held command buffer's own completed handler is the bound: it is the last of its
            // handlers to run, so once it has, the log is whole and the status is settled.
            log.wait_for(COMPLETED, COMPLETION);
            let handlers_at_the_end = log.names();
            let held_at_the_end = if handlers_at_the_end.iter().any(|name| name == COMPLETED) {
                Status::read(status_word(held.status()))
            } else {
                Status::NeverSettled
            };
            let behind_at_the_end = settled(&behind_done, &behind);
            Ok(HeldCommandBuffer {
                other_queue: other_queue_status,
                held_while_waiting,
                behind_while_waiting,
                handlers_while_waiting,
                handlers_at_the_end,
                held_at_the_end,
                behind_at_the_end,
            })
        })
    }

    fn signal_before_a_wait(&mut self) -> Answer<crate::event_orders::SignalBeforeWait> {
        use crate::event_orders::{Status, HELD_FOR, SCRATCH, SIGNAL_WAIT};
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let event = || {
                self.device
                    .newSharedEvent()
                    .ok_or_else(|| Failure::named("NewSharedEvent", "returned nil"))
            };
            let (before, gate, after) = (event()?, event()?, event()?);
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let fill = |byte: u8| -> Result<(), Failure> {
                let encoder = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
                encoder.fillBuffer_range_value(
                    &scratch,
                    objc2_foundation::NSRange::new(0, SCRATCH as usize),
                    byte,
                );
                encoder.endEncoding();
                Ok(())
            };
            let before_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*before);
            let gate_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*gate);
            let after_event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*after);
            fill(1)?;
            command_buffer.encodeSignalEvent_value(before_event, 1);
            command_buffer.encodeWaitForEvent_value(gate_event, 1);
            fill(2)?;
            command_buffer.encodeSignalEvent_value(after_event, 1);
            let done = completion_channel(&command_buffer);
            command_buffer.commit();
            let before_landed =
                before.waitUntilSignaledValue_timeoutMS(1, SIGNAL_WAIT.as_millis() as u64);
            // The window the two negative facts are false in: the signal behind the wait has not
            // landed, and the command buffer is no further along than committed.
            std::thread::sleep(HELD_FOR);
            let after_while_pending = after.signaledValue();
            let status_while_pending = Status::read(status_word(command_buffer.status()));
            gate.setSignaledValue(1);
            let status_at_the_end = settled(&done, &command_buffer);
            Ok(crate::event_orders::SignalBeforeWait {
                before_landed,
                after_while_pending,
                status_while_pending,
                after_at_the_end: after.signaledValue(),
                status_at_the_end,
            })
        })
    }

    fn scheduled_handler_order(
        &mut self,
    ) -> Answer<crate::scheduled_handlers::ScheduledHandlerOrder> {
        use crate::event_orders::{Status, COMPLETED, SCHEDULED, SCRATCH};
        use crate::scheduled_handlers::{
            Returned, ScheduledHandlerOrder, ARRIVAL, HOLD_LIMIT, WINDOW,
        };
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{mpsc, Arc, Mutex};
        objc2::rc::autoreleasepool(|_| {
            let queue = self
                .device
                .newCommandQueue()
                .ok_or_else(|| Failure::named("NewCommandQueue", "returned nil"))?;
            let scratch = self
                .device
                .newBufferWithLength_options(
                    SCRATCH as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or_else(|| Failure::named("NewBuffer", "returned nil"))?;
            let command_buffer = queue
                .commandBuffer()
                .ok_or_else(|| Failure::named("CommandBuffer", "returned nil"))?;
            let encoder = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil"))?;
            encoder.fillBuffer_range_value(
                &scratch,
                objc2_foundation::NSRange::new(0, SCRATCH as usize),
                1,
            );
            encoder.endEncoding();
            let order: Arc<Mutex<Vec<String>>> = Arc::default();
            let (entered, entering) = mpsc::channel();
            let (release, released) = mpsc::channel::<()>();
            let held = {
                let order = Arc::clone(&order);
                block2::RcBlock::new(
                    move |own: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        if let Ok(mut order) = order.lock() {
                            order.push(SCHEDULED.to_owned());
                        }
                        // SAFETY: Metal hands the handler its own command buffer, which is alive
                        // for the call.
                        let status = unsafe { own.as_ref() }.status();
                        let _ = entered.send(status_word(status).to_owned());
                        // Held, so the probe can read the status and the wait while this handler
                        // is inside itself. Bounded, so the handler leaves Metal's handler thread
                        // whatever the probe does.
                        let _ = released.recv_timeout(HOLD_LIMIT);
                    },
                )
            };
            let (done, completing) = mpsc::channel();
            let finished = {
                let order = Arc::clone(&order);
                block2::RcBlock::new(
                    move |own: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        if let Ok(mut order) = order.lock() {
                            order.push(COMPLETED.to_owned());
                        }
                        // SAFETY: as above.
                        let status = unsafe { own.as_ref() }.status();
                        let _ = done.send(status_word(status).to_owned());
                    },
                )
            };
            // SAFETY: both blocks are valid heap blocks held in `held` and `finished` for the rest
            // of this probe, which outlives every call Metal makes of them: the probe does not
            // return before the completed handler ran or its bound ran out.
            unsafe {
                command_buffer.addScheduledHandler(block2::RcBlock::as_ptr(&held));
                command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&finished));
            }
            command_buffer.commit();
            let inside_the_handler = match entering.recv_timeout(ARRIVAL) {
                Ok(word) => Status::read(word),
                Err(_) => Status::NeverSettled,
            };
            let while_the_handler_ran = Status::read(status_word(command_buffer.status()));
            let returned = Arc::new(AtomicBool::new(false));
            let (answered, answer) = mpsc::channel();
            {
                let (waited, returned) = (Waited(command_buffer.clone()), Arc::clone(&returned));
                // The wait runs on a thread of its own because the fact under test is that it does
                // **not** return; the thread is left to finish rather than joined, so a wait that
                // never came back leaves that thread and not the suite.
                std::thread::spawn(move || {
                    waited.wait_until_scheduled();
                    returned.store(true, Ordering::SeqCst);
                    let _ = answered.send(());
                });
            }
            // The window the negative fact is false in.
            std::thread::sleep(WINDOW);
            let wait_returned_while_the_handler_ran = returned.load(Ordering::SeqCst);
            let _ = release.send(());
            // Metal's `waitUntilScheduled` answers nothing, so a return is the whole answer.
            let wait_answered = match answer.recv_timeout(ARRIVAL) {
                Ok(()) => Returned::Ok,
                Err(_) => Returned::NotWithinTheWindow,
            };
            let inside_the_completed_handler = match completing.recv_timeout(ARRIVAL) {
                Ok(word) => Status::read(word),
                Err(_) => Status::NeverSettled,
            };
            let handler_order = order.lock().map(|order| order.clone()).unwrap_or_default();
            Ok(ScheduledHandlerOrder {
                inside_the_handler,
                while_the_handler_ran,
                wait_returned_while_the_handler_ran,
                wait_answered,
                inside_the_completed_handler,
                handler_order,
            })
        })
    }

    /// A texture over one plane of a real IOSurface laid out as the case says its stand-in buffer
    /// is (`newTextureWithDescriptor:iosurface:plane:`), or over the buffer itself.
    ///
    /// The surface is the only way Metal makes a texture of a two-plane YCbCr format at all: the
    /// descriptor alone answers nil for every one of the 69, which is what
    /// `ycbcr-two-plane-format-descriptors` records. So this is where the buffer the case states
    /// becomes the thing Metal takes — same plane offsets, same row pitches, same bytes from byte
    /// 0 — and the surface is kept beside the texture so it outlives it.
    fn plane_texture(
        &mut self,
        buffer: Handle,
        over: &crate::case::PlaneTexture,
    ) -> Answer<Handle> {
        let bytes = {
            let held = self.held(buffer)?;
            let length = held.length();
            // SAFETY: the buffer's own shared storage, whole, and no command uses it yet.
            unsafe {
                std::slice::from_raw_parts(held.contents().as_ptr().cast::<u8>(), length).to_vec()
            }
        };
        let [width, height] = over.size;
        // The descriptor and the texture are autoreleased, and a case may make several over a 24
        // MiB surface, so they are made inside a pool rather than left to a thread that drains
        // none.
        let made = objc2::rc::autoreleasepool(|_| {
            let descriptor = MTLTextureDescriptor::new();
            descriptor.setTextureType(MTLTextureType::Type2D);
            descriptor.setPixelFormat(MTLPixelFormat(over.format as usize));
            // SAFETY: the case's own extent; the setters read nothing but the value given.
            unsafe {
                descriptor.setWidth(width as usize);
                descriptor.setHeight(height as usize);
            }
            descriptor.setStorageMode(MTLStorageMode::Shared);
            match over.ostype {
                Some(ostype) => {
                    // The usage the runtime side's plane texture takes: a surface's plane is read.
                    descriptor.setUsage(MTLTextureUsage::ShaderRead);
                    let format = PixelFormat::try_from(over.format).map_err(|problem| {
                        Failure::named("NoSuchPixelFormat", format!("{}: {problem}", over.format))
                    })?;
                    let planes = crate::metal_surface::planes_of(format, over.size, over.planes)
                        .map_err(|why| Failure::named("NoSuchPlaneLayout", why))?;
                    crate::metal_surface::planes_hold_their_texels(&planes, bytes.len())
                        .map_err(|why| Failure::named("PlanesPastTheBuffer", why))?;
                    let surface = crate::metal_surface::Surface::two_plane(
                        ostype,
                        over.size,
                        &planes,
                        bytes.len(),
                    )
                    .map_err(|why| Failure::named("NoSurface", why))?;
                    surface
                        .planes_are_as_asked(&planes)
                        .map_err(|why| Failure::named("SurfaceLaidOtherwise", why))?;
                    surface
                        .fill(&bytes)
                        .map_err(|why| Failure::named("SurfaceNotFilled", why))?;
                    let device: &ProtocolObject<dyn MTLDevice> = &self.device;
                    let descriptor: &MTLTextureDescriptor = &descriptor;
                    let plane = over.plane as usize;
                    require_selector(
                        device,
                        objc2::sel!(newTextureWithDescriptor:iosurface:plane:),
                        "MTLDevice newTextureWithDescriptor:iosurface:plane:",
                    )?;
                    // SAFETY: `newTextureWithDescriptor:iosurface:plane:` of a live surface whose
                    // planes it states, at one of them; the surface is kept below, so it outlives
                    // the texture that reads it.
                    let texture: Option<Retained<ProtocolObject<dyn MTLTexture>>> = unsafe {
                        objc2::msg_send![device, newTextureWithDescriptor: descriptor, iosurface: surface.as_ptr(), plane: plane]
                    };
                    let texture = texture.ok_or_else(|| {
                        Failure::named(
                            "NewTextureWithIOSurfacePlane",
                            format!(
                                "format {} over plane {} of a {width}x{height} '{}' surface \
                                 returned nil",
                                over.format,
                                over.plane,
                                crate::metal_surface::four_cc(ostype)
                            ),
                        )
                    })?;
                    Ok((texture, Some(surface)))
                }
                None => {
                    // The usage the runtime side's texture over a buffer takes.
                    descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
                    let (offset, row) = over.planes[0];
                    let held = self.held(buffer)?;
                    let texture = held
                        .newTextureWithDescriptor_offset_bytesPerRow(
                            &descriptor,
                            offset as usize,
                            row as usize,
                        )
                        .ok_or_else(|| {
                            Failure::named(
                                "NewTextureWithDescriptorOffsetBytesPerRow",
                                format!(
                                    "format {} {width}x{height} at {offset} of {row} a row \
                                     returned nil",
                                    over.format
                                ),
                            )
                        })?;
                    Ok((texture, None))
                }
            }
        })?;
        let (texture, surface) = made;
        if let Some(surface) = surface {
            self.surfaces.push(surface);
        }
        Ok(keep(&mut self.textures, texture))
    }

    fn buffer_texture(
        &mut self,
        request: &crate::buffer_texture::BufferTextureRequest,
    ) -> Answer<crate::buffer_texture::BufferTextureFacts> {
        // The guard first, and before any object is made: a row pitch below the texture's own row
        // fails AGX's `bytes_per_row >= used_bytes_per_row` and ends this process, so the call is
        // never made. A case of that shape says `MetalSource::did_not_return` and record mode
        // refuses it; this is the second line, for a row somebody adds to a recorded table by
        // mistake.
        if let Some(why) = crate::buffer_texture::ends_metals_process(request) {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let (_, options) = storage_mode(request.buffer_storage);
        let buffer = self
            .device
            .newBufferWithLength_options(request.buffer_length as usize, options)
            .ok_or_else(|| {
                Failure::named(
                    "NewBufferWithLength",
                    format!("{} bytes returned nil", request.buffer_length),
                )
            })?;
        let descriptor = queried_descriptor(&request.descriptor);
        let texture = buffer
            .newTextureWithDescriptor_offset_bytesPerRow(
                &descriptor,
                request.offset as usize,
                request.bytes_per_row as usize,
            )
            .ok_or_else(|| {
                // Metal's own nil: the descriptor and the three numbers describe no texture over
                // the buffer, and the process goes on. The runtime answers `Error::Nil` for the
                // same shape, so the two sides say the same thing in their own spellings.
                Failure::named(
                    "Nil",
                    format!(
                        "newTextureWithDescriptor:offset:{}:bytesPerRow:{} returned nil",
                        request.offset, request.bytes_per_row
                    ),
                )
            })?;
        let facts = buffer_texture_facts(&texture);
        // The buffer and the texture are kept, so neither is freed while the sweep goes on and a
        // later row cannot read memory a dropped texture handed back.
        keep(&mut self.buffers, buffer);
        keep(&mut self.textures, texture);
        Ok(facts)
    }

    fn makes_texture(&mut self, query: &HeapTextureQuery) -> Answer<bool> {
        let descriptor = queried_descriptor(query);
        // The texture, where Metal makes one, is released as this statement ends: the case asks
        // this only of descriptors Metal makes none of, and nothing here keeps a handle, so no
        // sweep of the family holds a texture it does not read.
        Ok(self.device.newTextureWithDescriptor(&descriptor).is_some())
    }

    fn sparse_tile_size(&mut self, query: &SparseTileQuery) -> Answer<TileSize> {
        let texture_type = MTLTextureType(query.texture_type.as_raw() as usize);
        let format = MTLPixelFormat(query.format.as_raw() as usize);
        let samples = query.samples as usize;
        // SAFETY: both getters read the three scalars they are given and answer an `MTLSize`.
        // Neither allocates, compiles or reaches the GPU, and neither takes an object.
        let size = unsafe {
            match query.page {
                None => self
                    .device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount(
                        texture_type,
                        format,
                        samples,
                    ),
                Some(page) => self
                    .device
                    .sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                        texture_type,
                        format,
                        samples,
                        metal_page(page),
                    ),
            }
        };
        Ok(TileSize {
            width: size.width as u64,
            height: size.height as u64,
            depth: size.depth as u64,
        })
    }

    fn sparse_tile_size_in_bytes(&mut self, page: QueryPage) -> Answer<u64> {
        Ok(match page {
            None => self.device.sparseTileSizeInBytes() as u64,
            Some(page) => {
                self.device
                    .sparseTileSizeInBytesForSparsePageSize(metal_page(page)) as u64
            }
        })
    }

    fn sparse_heap(&mut self, spec: &SparseHeapSpec) -> Answer<Handle> {
        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Sparse);
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setSparsePageSize(metal_page(spec.page));
        descriptor.setSize(spec.size as usize);
        let heap = self
            .device
            .newHeapWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewHeapWithDescriptor",
                    format!("a sparse heap of {} bytes returned nil", spec.size),
                )
            })?;
        self.heaps.push(heap);
        Ok(Handle(self.heaps.len() as u32 - 1))
    }

    fn heap_facts(&mut self, heap: Handle) -> Answer<HeapFacts> {
        let heap = self.held_heap(heap)?;
        Ok(HeapFacts {
            heap_type: heap.r#type().0 as u64,
            size: heap.size() as u64,
            used_size: heap.usedSize() as u64,
        })
    }

    fn heap_texture(&mut self, heap: Handle, spec: &SparseTextureSpec) -> Answer<Handle> {
        let descriptor = sparse_texture_descriptor(spec);
        let texture = self
            .held_heap(heap)?
            .newTextureWithDescriptor(&descriptor)
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptor",
                    format!("{} on a heap returned nil", spec.site()),
                )
            })?;
        self.textures.push(texture);
        Ok(Handle(self.textures.len() as u32 - 1))
    }

    fn sparse_texture_facts(&mut self, texture: Handle) -> Answer<SparseTextureFacts> {
        let texture = self.held_texture(texture)?;
        Ok(SparseTextureFacts {
            is_sparse: texture.isSparse(),
            first_mipmap_in_tail: texture.firstMipmapInTail() as u64,
            tail_size_in_bytes: texture.tailSizeInBytes() as u64,
        })
    }
}

/// Metal's `MTLSparsePageSize` for a case's page, by its raw value, so the two spellings cannot
/// drift apart.
fn metal_page(page: SparsePageSize) -> MTLSparsePageSize {
    MTLSparsePageSize(page.raw() as isize)
}

/// Metal's descriptor for a case's sparse texture: every field the case names, by its raw ordinal.
///
/// The storage is the heap's (`Private`) and the usage is shader-read, with the render-target bit
/// where the case asks for it. Metal stops any other storage for a heap's texture even without its
/// validation layer, so neither is a field a case may set.
fn sparse_texture_descriptor(spec: &SparseTextureSpec) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType(spec.texture_type.as_raw() as usize));
    descriptor.setPixelFormat(MTLPixelFormat(spec.format.as_raw() as usize));
    // SAFETY: each extent and count is a case's own, and the setters read nothing but the value
    // they are given.
    unsafe {
        descriptor.setWidth(spec.size[0] as usize);
        descriptor.setHeight(spec.size[1] as usize);
        descriptor.setDepth(spec.size[2] as usize);
        descriptor.setMipmapLevelCount(spec.levels as usize);
        descriptor.setSampleCount(spec.samples as usize);
        descriptor.setArrayLength(spec.array_length as usize);
    }
    let usage = if spec.render_target {
        MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget
    } else {
        MTLTextureUsage::ShaderRead
    };
    descriptor.setUsage(usage);
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor
}

/// A command buffer the slot probe holds, so the thread that frees a slot need not be the thread
/// that made it.
///
/// `objc2` gives `MTLCommandBuffer` no `Send`, because Metal documents a command buffer as unsafe
/// to *encode into* from more than one thread. The probe encodes into none: it makes each command
/// buffer on one thread and only ever releases it from another, and releasing an Objective-C object
/// off the thread that made it is sound, since the retain count is atomic.
pub struct Made(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: as the type's documentation states: the probe never encodes into a held command buffer,
// and the only operation that crosses threads is the release.
unsafe impl Send for Made {}

impl MetalSide {
    /// The plane behind the texture `target`.
    fn held_plane(&self, target: Handle) -> Answer<(&crate::metal_surface::Surface, usize)> {
        self.planes
            .iter()
            .find(|plane| plane.handle == target)
            .map(|plane| (&self.surfaces[plane.at], plane.length))
            .ok_or_else(|| {
                Failure::named(
                    "NotOverASurface",
                    format!("texture {} lies over no IOSurface", target.0),
                )
            })
    }

    /// A framebuffer-only texture of `spec`'s format and extent over one plane of an IOSurface this
    /// side makes for it, at `plane`'s rotation and stride.
    ///
    /// Three things here are not ordinary Metal, and each is guarded:
    ///
    /// - **The stride rule is a safety rule.** Metal asserts, *ending the process*, on an IOSurface
    ///   texture whose `bytesPerRow` is below the texture's own row
    ///   ([`crate::surface`]), so the plane is refused here before any descriptor reaches Metal.
    /// - **`setFramebufferOnly:` and `setRotation:` are Apple SPI.** Both are reached by selector
    ///   and only after `respondsToSelector:`, so a release that stops answering them gives a typed
    ///   refusal naming the selector instead of an unrecognised-selector crash.
    /// - **`newTextureWithDescriptor:iosurface:plane:` is public**, but its `IOSurfaceRef`
    ///   argument belongs to a framework this crate does not depend on, so it is sent by selector
    ///   with the raw pointer.
    ///
    /// A target as a texture over a buffer the case made:
    /// `newTextureWithDescriptor:offset:bytesPerRow:`.
    ///
    /// The stride is checked **before any object is made**, because a `bytesPerRow` below the
    /// target's own row ends Metal's process rather than answering nil
    /// ([`crate::render::OverBuffer::unsafe_stride`]). That is the same order the surface target
    /// takes for the same reason, and it is what lets a case that would stop the oracle be
    /// refused on a machine with no device at all.
    fn buffer_backed_target(
        &mut self,
        spec: &TargetSpec,
        over: &crate::render::OverBuffer,
    ) -> Answer<Handle> {
        if let Some(why) = over.unsafe_stride() {
            return Err(Failure::named("WouldEndMetalsProcess", why));
        }
        let buffer = self.held(over.buffer)?;
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        // A texture over a buffer holds the buffer's own bytes, so its storage is the buffer's.
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = buffer
            .newTextureWithDescriptor_offset_bytesPerRow(
                &descriptor,
                over.offset as usize,
                over.bytes_per_row as usize,
            )
            .ok_or_else(|| {
                Failure::named(
                    "NewTextureWithDescriptorOffsetBytesPerRow",
                    format!(
                        "a {:?} {}x{} target over a buffer at offset {} and {} bytes a row \
                         returned nil",
                        spec.format, spec.size[0], spec.size[1], over.offset, over.bytes_per_row
                    ),
                )
            })?;
        Ok(keep(&mut self.textures, texture))
    }

    /// Metal answering nil is a refusal, which is safe and is what the suite reads a venue's gap
    /// out of.
    fn surface_target(
        &mut self,
        spec: &TargetSpec,
        plane: &crate::surface::PlaneSpec,
    ) -> Answer<Handle> {
        if plane.unsafe_stride(spec.size) {
            return Err(Failure::named(
                "WouldEndMetalsProcess",
                format!(
                    "a plane of {} bytes a row behind a {} texel row: Metal's descriptor \
                     validation asserts (\"IOSurface texture: bytesPerRow must be greater or equal \
                     to\"), which ends the process",
                    plane.row_pitch,
                    spec.size[0] * plane.texel
                ),
            ));
        }
        // SAFETY: the descriptor constructor reads only the format and the size it is given.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat(spec.format.as_raw() as usize),
                spec.size[0] as usize,
                spec.size[1] as usize,
                false,
            )
        };
        let mut usage = MTLTextureUsage::RenderTarget;
        if spec.shader_read {
            usage |= MTLTextureUsage::ShaderRead;
        }
        if spec.shader_write {
            usage |= MTLTextureUsage::ShaderWrite;
        }
        descriptor.setUsage(usage);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let framebuffer_only = objc2::sel!(setFramebufferOnly:);
        let rotation = objc2::sel!(setRotation:);
        for (selector, name) in [
            (
                framebuffer_only,
                "MTLTextureDescriptor setFramebufferOnly: (Apple SPI)",
            ),
            (rotation, "MTLTextureDescriptor setRotation: (Apple SPI)"),
        ] {
            require_selector(&*descriptor, selector, name)?;
        }
        // SAFETY: both selectors were just checked, and each takes one integral argument and
        // answers nothing, as the private setters do. Verified answered on an Apple M3 Max under
        // macOS 27.0.1, where the descriptor also reads the rotation back.
        unsafe {
            let _: () = objc2::msg_send![&*descriptor, setFramebufferOnly: true];
            let _: () = objc2::msg_send![&*descriptor, setRotation: plane.rotation as objc2_foundation::NSUInteger];
        }
        let surface = crate::metal_surface::Surface::single_plane(
            plane.plane,
            plane.texel,
            plane.row_pitch,
            plane.pad,
        )
        .map_err(|why| Failure::named("IOSurfaceCreate", why))?;
        // The layout was a request, not an instruction, so it is read back before a texture lies
        // over it: a surface whose rows are not where the case says they are would make every
        // recorded byte a comparison at the wrong offset.
        surface
            .is_as_asked(plane.plane, plane.row_pitch)
            .map_err(|why| Failure::named("SurfaceNotAsAsked", why))?;
        let texture = self.texture_over_surface(&descriptor, &surface, plane)?;
        let handle = keep(&mut self.textures, texture);
        let at = self.surfaces.len();
        self.surfaces.push(surface);
        self.planes.push(PlaneAt {
            handle,
            at,
            length: plane.length(),
        });
        Ok(handle)
    }

    /// `newTextureWithDescriptor:iosurface:plane:` of plane 0, by selector.
    ///
    /// The plane's bytes are already the pad byte: [`crate::metal_surface::Surface::single_plane`]
    /// sets them, so there is no window in which Metal could read a surface this side had not
    /// written.
    fn texture_over_surface(
        &self,
        descriptor: &MTLTextureDescriptor,
        surface: &crate::metal_surface::Surface,
        plane: &crate::surface::PlaneSpec,
    ) -> Answer<Retained<ProtocolObject<dyn MTLTexture>>> {
        let selector = objc2::sel!(newTextureWithDescriptor:iosurface:plane:);
        require_selector(
            &*self.device,
            selector,
            "MTLDevice newTextureWithDescriptor:iosurface:plane:",
        )?;
        // SAFETY: the selector was just checked; it takes the descriptor, an `IOSurfaceRef` and a
        // plane index, and answers a `+1` texture, which `Retained` takes ownership of. The stride
        // rule the caller checked is what keeps this call from asserting.
        let texture: Option<Retained<ProtocolObject<dyn MTLTexture>>> = unsafe {
            objc2::msg_send![
                &*self.device,
                newTextureWithDescriptor: descriptor,
                iosurface: surface.as_ptr(),
                plane: 0usize,
            ]
        };
        texture.ok_or_else(|| {
            Failure::named("NewTextureWithIoSurface", format!("{plane:?} returned nil"))
        })
    }
}

/// Check `object` answers `selector` before anything is sent to it, answering
/// [`Failure::unavailable`] naming the selector where it does not.
///
/// **Every Apple SPI this suite sends goes through here**: the two texture fills
/// ([`crate::metal_work`]), `MTLTextureDescriptor`'s `setFramebufferOnly:` and `setRotation:`, and
/// the `IOSurfaceRef` creator. One lookup rather than one per call site, so a release that stops
/// answering a private selector gives the same typed refusal naming it wherever it is reached,
/// instead of an unrecognised-selector crash in whichever module happened to forget the check.
pub(crate) fn require_selector(
    object: &(impl NSObjectProtocol + objc2::Message),
    selector: objc2::runtime::Sel,
    name: &'static str,
) -> Answer<()> {
    match object.respondsToSelector(selector) {
        true => Ok(()),
        false => Err(Failure::unavailable(name)),
    }
}

/// Metal's descriptor for a queried one: every field the case names, by its raw ordinal, so a
/// private format and a private usage bit reach Metal as readily as a public one.
fn queried_descriptor(query: &HeapTextureQuery) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    descriptor.setTextureType(MTLTextureType(query.texture_type.as_raw() as usize));
    descriptor.setPixelFormat(MTLPixelFormat(query.format.as_raw() as usize));
    // SAFETY: each extent, count and index is a case's own, and the setters read nothing but the
    // value they are given.
    unsafe {
        descriptor.setWidth(query.size[0] as usize);
        descriptor.setHeight(query.size[1] as usize);
        descriptor.setDepth(query.size[2] as usize);
        descriptor.setMipmapLevelCount(query.levels as usize);
        descriptor.setSampleCount(query.samples as usize);
        descriptor.setArrayLength(query.array_length as usize);
    }
    descriptor.setUsage(MTLTextureUsage(query.usage.as_raw() as usize));
    descriptor.setStorageMode(storage_mode(query.storage).0);
    descriptor
}

/// What a texture Metal laid over a buffer answers of itself: every getter the runtime's `Texture`
/// has too, so a difference names the field rather than the object.
fn buffer_texture_facts(
    texture: &ProtocolObject<dyn MTLTexture>,
) -> crate::buffer_texture::BufferTextureFacts {
    crate::buffer_texture::BufferTextureFacts {
        texture_type: texture.textureType().0 as u64,
        pixel_format: texture.pixelFormat().0 as u64,
        width: texture.width() as u64,
        height: texture.height() as u64,
        depth: texture.depth() as u64,
        mipmap_level_count: texture.mipmapLevelCount() as u64,
        array_length: texture.arrayLength() as u64,
        sample_count: texture.sampleCount() as u64,
        usage: texture.usage().0 as u64,
        storage_mode: texture.storageMode().0 as u64,
        cpu_cache_mode: texture.cpuCacheMode().0 as u64,
        hazard_tracking_mode: texture.hazardTrackingMode().0 as u64,
        gpu_optimized: texture.allowGPUOptimizedContents(),
    }
}

/// The case's enumerations reach Metal's by their raw ordinals, which are Metal's own.
fn compare_function(function: metal2vulkan::metal::CompareFunction) -> MTLCompareFunction {
    MTLCompareFunction(function.as_raw() as usize)
}

fn stencil_operation(operation: metal2vulkan::metal::StencilOperation) -> MTLStencilOperation {
    MTLStencilOperation(operation.as_raw() as usize)
}

/// One face's stencil state as Metal's descriptor.
fn stencil_descriptor(face: &StencilDescriptor) -> Retained<MTLStencilDescriptor> {
    let descriptor = MTLStencilDescriptor::new();
    descriptor.setStencilCompareFunction(compare_function(face.stencil_compare_function));
    descriptor.setStencilFailureOperation(stencil_operation(face.stencil_failure_operation));
    descriptor.setDepthFailureOperation(stencil_operation(face.depth_failure_operation));
    descriptor.setDepthStencilPassOperation(stencil_operation(face.depth_stencil_pass_operation));
    descriptor.setReadMask(face.read_mask);
    descriptor.setWriteMask(face.write_mask);
    descriptor
}

/// The storage mode and resource options a [`Storage`] names: the Metal side's one spelling of the
/// three modes, which [`crate::metal_work`] reads too, so a case's storage mode cannot mean one
/// thing to a queried descriptor and another to a program's buffer.
pub fn storage_mode(storage: Storage) -> (MTLStorageMode, MTLResourceOptions) {
    match storage {
        Storage::Shared => (
            MTLStorageMode::Shared,
            MTLResourceOptions::StorageModeShared,
        ),
        Storage::Managed => (
            MTLStorageMode::Managed,
            MTLResourceOptions::StorageModeManaged,
        ),
        Storage::Private => (
            MTLStorageMode::Private,
            MTLResourceOptions::StorageModePrivate,
        ),
    }
}

/// The pixel format [`crate::case::TARGET_FORMAT`] names, as Metal spells it.
const TARGET_PIXEL_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA32Uint;

impl MetalSide {
    /// The descriptor of a mesh render pipeline of `library`'s `functions`.
    fn mesh_descriptor(
        &self,
        library: Handle,
        functions: &MeshFunctions<'_>,
    ) -> Answer<Retained<MTLMeshRenderPipelineDescriptor>> {
        let library = self
            .libraries
            .get(library.0 as usize)
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("library {}", library.0)))?;
        let function = |name: &str| {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| Failure::named("NoSuchFunction", name.to_owned()))
        };
        let descriptor = MTLMeshRenderPipelineDescriptor::new();
        // SAFETY: each function is of this library, and the descriptor is this call's own.
        unsafe {
            descriptor.setMeshFunction(Some(&*function(functions.mesh)?));
            if let Some(object) = functions.object {
                descriptor.setObjectFunction(Some(&*function(object)?));
            }
            descriptor.setFragmentFunction(Some(&*function(functions.fragment)?));
            let attachments = descriptor.colorAttachments();
            for at in 0..functions.attachments as usize {
                attachments
                    .objectAtIndexedSubscript(at)
                    .setPixelFormat(TARGET_PIXEL_FORMAT);
            }
        }
        Ok(descriptor)
    }

    /// The state `descriptor` makes, kept by handle.
    fn made_mesh_pipeline(
        &mut self,
        descriptor: &MTLMeshRenderPipelineDescriptor,
    ) -> Answer<Handle> {
        let state = self
            .device
            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|error| Failure::named("NewMeshRenderPipelineState", error.to_string()))?;
        Ok(keep(
            &mut self.render_pipelines,
            RenderPipeline::unreflected(state),
        ))
    }

    /// One command buffer and one render encoder over `targets`, each attachment cleared to zero
    /// and stored.
    #[allow(clippy::type_complexity)]
    fn pass(
        &self,
        targets: &[Handle],
    ) -> Answer<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    )> {
        let attachments: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, target) in attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments every Metal pass holds.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(target));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        Ok((command_buffer, encoder))
    }

    /// One command buffer and one render encoder over `targets` in `tile`-sized tiles, each
    /// attachment loaded and stored.
    #[allow(clippy::type_complexity)]
    fn loading_pass(
        &self,
        targets: &[Handle],
        tile: [u32; 2],
    ) -> Answer<(
        Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    )> {
        let attachments: Vec<Retained<ProtocolObject<dyn MTLTexture>>> = targets
            .iter()
            .map(|handle| self.held_texture(*handle).map(Retained::from))
            .collect::<Answer<Vec<_>>>()?;
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, target) in attachments.iter().enumerate() {
            // SAFETY: `at` is below the eight attachments every Metal pass holds.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(target));
            attachment.setLoadAction(MTLLoadAction::Load);
            attachment.setStoreAction(MTLStoreAction::Store);
        }
        pass.setTileWidth(tile[0] as usize);
        pass.setTileHeight(tile[1] as usize);
        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Failure::named("CommandBuffer", "returned nil".to_owned()))?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        Ok((command_buffer, encoder))
    }

    /// Encode one of a case's tile-pass steps into `encoder`.
    fn encode_tile_step(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &TileStep,
    ) -> Answer<()> {
        match step {
            TileStep::Pipeline(state) => {
                let state = &self
                    .render_pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("tile pipeline {}", state.0))
                    })?
                    .state;
                encoder.setRenderPipelineState(state);
            }
            TileStep::TileBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // encoder outlives the call.
                unsafe {
                    encoder.setTileBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            TileStep::TileBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them.
                unsafe {
                    encoder.setTileBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            TileStep::TileTexture { texture, index } => {
                let texture = self.held_texture(*texture)?;
                // SAFETY: the texture is kept alive by `self.textures`, and the encoder outlives
                // the call.
                unsafe { encoder.setTileTexture_atIndex(Some(texture), *index as usize) };
            }
            TileStep::DispatchThreadsPerTile(threads) => {
                encoder.dispatchThreadsPerTile(MTLSize {
                    width: threads[0] as usize,
                    height: threads[1] as usize,
                    depth: threads[2] as usize,
                });
            }
        }
        Ok(())
    }

    /// Encode one of a case's render-pass steps into `encoder`.
    fn encode_step(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &MeshStep,
    ) -> Answer<()> {
        let size = |axes: [u32; 3]| MTLSize {
            width: axes[0] as usize,
            height: axes[1] as usize,
            depth: axes[2] as usize,
        };
        match step {
            MeshStep::Pipeline(state) => {
                let state = &self
                    .render_pipelines
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("mesh pipeline {}", state.0))
                    })?
                    .state;
                encoder.setRenderPipelineState(state);
            }
            MeshStep::ObjectBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the function's binding, and the
                // encoder outlives the call.
                unsafe {
                    encoder.setObjectBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::ObjectBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them.
                unsafe {
                    encoder.setObjectBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            MeshStep::MeshBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe {
                    encoder.setMeshBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: as the object buffer's.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(buffer),
                        *offset as usize,
                        *index as usize,
                    )
                };
            }
            MeshStep::DrawThreadgroups {
                groups,
                object,
                mesh,
            } => encoder
                .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(*groups),
                    size(*object),
                    size(*mesh),
                ),
            MeshStep::DrawThreads {
                threads,
                object,
                mesh,
            } => encoder.drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                size(*threads),
                size(*object),
                size(*mesh),
            ),
            MeshStep::DrawIndirect {
                buffer,
                offset,
                object,
                mesh,
            } => {
                let buffer = self.held(*buffer)?;
                // SAFETY: the buffer holds three 32-bit counts at the offset, which the case
                // wrote before the pass.
                unsafe {
                    encoder.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        buffer,
                        *offset as usize,
                        size(*object),
                        size(*mesh),
                    );
                }
            }
            MeshStep::BarrierScope {
                scope,
                after,
                before,
            } => encoder.memoryBarrierWithScope_afterStages_beforeStages(
                MTLBarrierScope(*scope as usize),
                MTLRenderStages(*after as usize),
                MTLRenderStages(*before as usize),
            ),
            MeshStep::BarrierResources {
                buffers,
                after,
                before,
            } => {
                let over: Vec<&ProtocolObject<dyn MTLResource>> = buffers
                    .iter()
                    .map(|handle| self.held(*handle).map(ProtocolObject::from_ref))
                    .collect::<Answer<Vec<_>>>()?;
                let mut pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> =
                    over.iter().map(|resource| (*resource).into()).collect();
                let first = std::ptr::NonNull::new(pointers.as_mut_ptr())
                    .ok_or_else(|| Failure::named("NoResources", "an empty barrier"))?;
                // SAFETY: `pointers` holds exactly `pointers.len()` resource pointers, each
                // retained by `over` for the whole call.
                unsafe {
                    encoder.memoryBarrierWithResources_count_afterStages_beforeStages(
                        first,
                        pointers.len(),
                        MTLRenderStages(*after as usize),
                        MTLRenderStages(*before as usize),
                    );
                }
            }
        }
        Ok(())
    }

    /// The samples of the pass's own attachments, which is the count Metal holds its custom
    /// sample positions to: the first attachment the pass states, and 1 for a pass with none.
    fn pass_samples(&self, pass: &RenderPass) -> Answer<u32> {
        let attached = pass
            .colors
            .iter()
            .filter_map(|color| color.texture)
            .chain(pass.depth.as_ref().map(|depth| depth.texture))
            .chain(pass.stencil.as_ref().map(|stencil| stencil.texture))
            .next();
        match attached {
            Some(handle) => Ok(self.held_texture(handle)?.sampleCount() as u32),
            None => Ok(1),
        }
    }

    /// Encode one render pass of a program into `command_buffer`.
    fn encode_pass(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        pass: &RenderPass,
    ) -> Answer<()> {
        let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
        for (at, color) in pass.colors.iter().enumerate() {
            let Some(texture) = color.texture else {
                continue;
            };
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(texture)?);
            // SAFETY: `at` is below the eight color attachments every Metal pass holds, which the
            // case's own guard keeps it to.
            let attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(at) };
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction(color.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(color.store.raw() as usize));
            attachment.setClearColor(MTLClearColor {
                red: color.clear[0],
                green: color.clear[1],
                blue: color.clear[2],
                alpha: color.clear[3],
            });
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                color.options.raw() as usize,
            ));
            if let Some(resolve) = color.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(depth) = &pass.depth {
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(depth.texture)?);
            let attachment = descriptor.depthAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction(depth.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(depth.store.raw() as usize));
            attachment.setClearDepth(depth.clear);
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                depth.options.raw() as usize,
            ));
            attachment.setDepthResolveFilter(MTLMultisampleDepthResolveFilter(
                depth.filter.raw() as usize
            ));
            if let Some(resolve) = depth.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(stencil) = &pass.stencil {
            let texture: Retained<ProtocolObject<dyn MTLTexture>> =
                Retained::from(self.held_texture(stencil.texture)?);
            let attachment = descriptor.stencilAttachment();
            attachment.setTexture(Some(&texture));
            attachment.setLoadAction(MTLLoadAction(stencil.load.raw() as usize));
            attachment.setStoreAction(MTLStoreAction(stencil.store.raw() as usize));
            attachment.setClearStencil(stencil.clear);
            attachment.setStoreActionOptions(MTLStoreActionOptions::from_bits_retain(
                stencil.options.raw() as usize,
            ));
            attachment.setStencilResolveFilter(MTLMultisampleStencilResolveFilter(
                stencil.filter.raw() as usize,
            ));
            if let Some(resolve) = stencil.resolve {
                let resolve: Retained<ProtocolObject<dyn MTLTexture>> =
                    Retained::from(self.held_texture(resolve)?);
                attachment.setResolveTexture(Some(&resolve));
            }
        }
        if let Some(buffer) = pass.visibility {
            descriptor.setVisibilityResultBuffer(Some(self.held(buffer)?));
        }
        if let Some(size) = pass.size {
            descriptor.setRenderTargetWidth(size[0] as usize);
            descriptor.setRenderTargetHeight(size[1] as usize);
            descriptor.setDefaultRasterSampleCount(1);
        }
        if !pass.positions.is_empty() {
            // The one place the Metal leg encodes `setSamplePositions:count:`, so the guard that
            // keeps the suite off Metal's fatal shapes is reached by every case that states
            // positions, whatever else it does. The samples are the pass's own attachments', which
            // is what Metal counts the positions against.
            let samples = self.pass_samples(pass)?;
            if let Some(refusal) = crate::render::forbidden_positions(&pass.positions, samples) {
                return Err(refusal);
            }
            let positions: Vec<MTLSamplePosition> = pass
                .positions
                .iter()
                .map(|position| MTLSamplePosition {
                    x: position.x,
                    y: position.y,
                })
                .collect();
            // SAFETY: the pointer is to `positions.len()` positions that live for the call, and
            // the count is at or below the pass's samples, which the guard above has just held.
            unsafe {
                descriptor.setSamplePositions_count(positions.as_ptr(), positions.len());
            }
        }
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&descriptor)
            .ok_or_else(|| Failure::named("RenderCommandEncoder", "returned nil".to_owned()))?;
        for step in &pass.steps {
            self.encode_step_of(&encoder, step)?;
        }
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one of a case's render-pass steps into `encoder`.
    fn encode_step_of(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        step: &RenderStep,
    ) -> Answer<()> {
        let primitive =
            |kind: metal2vulkan::metal::PrimitiveType| MTLPrimitiveType(kind.as_raw() as usize);
        match step {
            RenderStep::Pipeline(state) => {
                let state = self.held_render_state(*state)?;
                encoder.setRenderPipelineState(&state);
            }
            RenderStep::DepthStencilState(state) => {
                let state = self
                    .depth_stencil_states
                    .get(state.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("depth/stencil state {}", state.0))
                    })?;
                encoder.setDepthStencilState(Some(state));
            }
            RenderStep::StencilReference(value) => encoder.setStencilReferenceValue(*value),
            RenderStep::VertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the vertex function's binding from
                // that offset, which the case states as one of its inputs, and the encoder
                // outlives the call.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(held),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: as the vertex buffer's, for the fragment function's binding.
                unsafe {
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(held),
                        *offset as usize,
                        *index as usize,
                    );
                }
            }
            RenderStep::FragmentTexture { texture, index } => {
                let held = self.held_texture(*texture)?;
                // SAFETY: the texture is of the kind and access the fragment function declares at
                // that slot, and the encoder outlives the call.
                unsafe { encoder.setFragmentTexture_atIndex(Some(held), *index as usize) };
            }
            RenderStep::FragmentSampler { sampler, index } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the fragment function declares a sampler at that slot, and the encoder
                // outlives the call.
                unsafe { encoder.setFragmentSamplerState_atIndex(Some(held), *index as usize) };
            }
            RenderStep::VertexBytes { bytes, index } => {
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the vertex function's binding at that slot.
                unsafe {
                    encoder.setVertexBytes_length_atIndex(pointer, bytes.len(), *index as usize)
                };
            }
            RenderStep::Viewport(rect) => encoder.setViewport(MTLViewport {
                originX: rect.origin[0],
                originY: rect.origin[1],
                width: rect.size[0],
                height: rect.size[1],
                znear: rect.depth[0],
                zfar: rect.depth[1],
            }),
            RenderStep::BlendColor(channels) => {
                let [red, green, blue, alpha] = channels.0;
                encoder.setBlendColorRed_green_blue_alpha(red, green, blue, alpha);
            }
            RenderStep::VisibilityResultMode { mode, offset } => {
                encoder.setVisibilityResultMode_offset(
                    MTLVisibilityResultMode(mode.raw() as usize),
                    *offset as usize,
                );
            }
            RenderStep::Draw {
                primitive: kind,
                first,
                count,
                instances,
            } => {
                // SAFETY: the draw names a vertex range of the case's own triangle, and every
                // buffer the vertex function reads was bound by a step before it.
                unsafe {
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        primitive(*kind),
                        *first as usize,
                        *count as usize,
                        *instances as usize,
                        0,
                    );
                }
            }
            RenderStep::DrawIndexed {
                primitive: kind,
                count,
                index_type,
                indices,
                offset,
                instances,
                base_vertex,
                base_instance,
            } => {
                let buffer = self.held(*indices)?;
                // SAFETY: the index buffer holds the case's own indices from that offset, which
                // the case states as one of its inputs, and the counts are its own.
                unsafe {
                    encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive(*kind),
                        *count as usize,
                        MTLIndexType(index_type.as_raw() as usize),
                        buffer,
                        *offset as usize,
                        *instances as usize,
                        *base_vertex as isize,
                        *base_instance as usize,
                    );
                }
            }
            RenderStep::DrawIndirect {
                primitive: kind,
                arguments,
                offset,
            } => {
                let buffer = self.held(*arguments)?;
                // SAFETY: the buffer holds the draw's four 32-bit arguments at that offset, which
                // the case wrote before the pass.
                unsafe {
                    encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                        primitive(*kind),
                        buffer,
                        *offset as usize,
                    );
                }
            }
        }
        Ok(())
    }
}

/// The render vocabulary ([`crate::render`]) on native Metal.
impl MetalSide {
    /// The render pipeline state `handle` names.
    fn held_render_state(
        &self,
        handle: Handle,
    ) -> Answer<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.render_pipelines
            .get(handle.0 as usize)
            .map(|held| held.state.clone())
            .ok_or_else(|| Failure::named("NoSuchHandle", format!("render pipeline {}", handle.0)))
    }

    /// Encode one compute dispatch of a program into `command_buffer`.
    fn encode_dispatch(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: &Dispatch,
    ) -> Answer<()> {
        let state = self
            .pipelines
            .get(dispatch.pipeline.0 as usize)
            .ok_or_else(|| {
                Failure::named(
                    "NoSuchHandle",
                    format!("compute pipeline {}", dispatch.pipeline.0),
                )
            })?
            .clone();
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| Failure::named("ComputeCommandEncoder", "returned nil".to_owned()))?;
        encoder.setComputePipelineState(&state);
        for (at, (buffer, offset)) in dispatch.buffers.iter().enumerate() {
            let held = self.held(*buffer)?;
            // SAFETY: the buffer is at least as long as the kernel's binding from that offset,
            // which the case states as one of its inputs, and the encoder outlives the call.
            unsafe { encoder.setBuffer_offset_atIndex(Some(held), *offset as usize, at) };
        }
        for (at, texture) in dispatch.textures.iter().enumerate() {
            let held = self.held_texture(*texture)?;
            // SAFETY: the texture is of the kind and access the kernel declares at that slot, and
            // the encoder outlives the call.
            unsafe { encoder.setTexture_atIndex(Some(held), at) };
        }
        encoder.dispatchThreads_threadsPerThreadgroup(size(dispatch.grid), size(dispatch.threads));
        encoder.endEncoding();
        Ok(())
    }

    /// Encode one command of a compute pass into `encoder`.
    ///
    /// Each arm is one Metal call, with the handles resolved through the side's own pools, so a
    /// case's command list reads as the calls it makes.
    fn encode_compute_step(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        step: &ComputeStep,
    ) -> Answer<()> {
        // A resource of either kind as `id<MTLResource>`: what the residency declarations and the
        // resource barrier take.
        let resource = |named: &PassResource| -> Answer<Retained<ProtocolObject<dyn MTLResource>>> {
            match named {
                PassResource::Buffer(handle) => {
                    let held = self.held(*handle)?;
                    Ok(Retained::from(ProtocolObject::<dyn MTLResource>::from_ref(
                        held,
                    )))
                }
                PassResource::Texture(handle) => {
                    let held = self.held_texture(*handle)?;
                    Ok(Retained::from(ProtocolObject::<dyn MTLResource>::from_ref(
                        held,
                    )))
                }
            }
        };
        match step {
            ComputeStep::Pipeline(pipeline) => {
                let state = self
                    .pipelines
                    .get(pipeline.0 as usize)
                    .ok_or_else(|| {
                        Failure::named("NoSuchHandle", format!("pipeline {}", pipeline.0))
                    })?
                    .clone();
                encoder.setComputePipelineState(&state);
            }
            ComputeStep::Buffer {
                buffer,
                offset,
                index,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer is at least as long as the kernel's binding from that
                // offset, which the case states as one of its inputs, and the encoder outlives
                // the call.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(held), *offset as usize, *index as usize)
                };
            }
            ComputeStep::BufferOffset { offset, index } => {
                // SAFETY: a buffer is bound at that index already — the case states the binding
                // before this command — and the offset leaves the kernel's binding inside it.
                unsafe { encoder.setBufferOffset_atIndex(*offset as usize, *index as usize) };
            }
            ComputeStep::Buffers { buffers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>> = buffers
                    .iter()
                    .map(|(handle, _)| self.held(*handle).map(|held| Some(Retained::from(held))))
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let offsets: Vec<usize> = buffers.iter().map(|(_, at)| *at as usize).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: both arrays hold the range's length of elements, each buffer outlives
                // the call, and each binding lies inside its buffer from its offset.
                unsafe {
                    encoder.setBuffers_offsets_withRange(
                        array_head(&pointers)?,
                        array_head(&offsets)?,
                        range,
                    );
                }
            }
            ComputeStep::Bytes { bytes, index } => {
                // `bytes.as_ptr()`, never `NonNull::from(bytes)`: the latter is the address of the
                // `Vec` itself — its pointer, capacity and length — and Metal would copy those
                // words into the kernel's binding. Written as the three inline bindings beside it
                // are (`setTileBytes:`, `setObjectBytes:`, `setVertexBytes:`); taking the other
                // one first made this case's own Metal leg read a kernel's coordinates out of a
                // `Vec` header, which read exactly like a product bug.
                let pointer =
                    std::ptr::NonNull::new(bytes.as_ptr().cast::<std::ffi::c_void>().cast_mut())
                        .ok_or_else(|| Failure::named("NoBytes", "an empty inline binding"))?;
                // SAFETY: `bytes` lives for the call, which copies them, and is at least as long
                // as the kernel's binding at that slot.
                unsafe { encoder.setBytes_length_atIndex(pointer, bytes.len(), *index as usize) };
            }
            ComputeStep::Texture { texture, index } => {
                let held = self.held_texture(*texture)?;
                // SAFETY: the texture is of the kind and access the kernel declares at that slot,
                // and the encoder outlives the call.
                unsafe { encoder.setTexture_atIndex(Some(held), *index as usize) };
            }
            ComputeStep::Textures { textures, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLTexture>>>> = textures
                    .iter()
                    .map(|handle| {
                        self.held_texture(*handle)
                            .map(|held| Some(Retained::from(held)))
                    })
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let range = range_of(*first, &pointers)?;
                // SAFETY: the array holds the range's length of pointers and each texture
                // outlives the call.
                unsafe { encoder.setTextures_withRange(array_head(&pointers)?, range) };
            }
            ComputeStep::Sampler { sampler, index } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: the kernel declares a sampler at that slot, and the encoder outlives
                // the call.
                unsafe { encoder.setSamplerState_atIndex(Some(held), *index as usize) };
            }
            ComputeStep::Samplers { samplers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLSamplerState>>>> = samplers
                    .iter()
                    .map(|handle| {
                        self.held_sampler(*handle)
                            .map(|held| Some(Retained::from(held)))
                    })
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let range = range_of(*first, &pointers)?;
                // SAFETY: as for `setTextures:withRange:`.
                unsafe { encoder.setSamplerStates_withRange(array_head(&pointers)?, range) };
            }
            ComputeStep::ClampedSampler {
                sampler,
                lod_min,
                lod_max,
                index,
            } => {
                let held = self.held_sampler(*sampler)?;
                // SAFETY: as for `setSamplerState:atIndex:`; the clamps are two floats Metal
                // reads by value.
                unsafe {
                    encoder.setSamplerState_lodMinClamp_lodMaxClamp_atIndex(
                        Some(held),
                        *lod_min,
                        *lod_max,
                        *index as usize,
                    );
                }
            }
            ComputeStep::ClampedSamplers { samplers, first } => {
                let held: Vec<Option<Retained<ProtocolObject<dyn MTLSamplerState>>>> = samplers
                    .iter()
                    .map(|(handle, _, _)| {
                        self.held_sampler(*handle)
                            .map(|held| Some(Retained::from(held)))
                    })
                    .collect::<Answer<Vec<_>>>()?;
                let pointers = pointer_array(&held);
                let mins: Vec<f32> = samplers.iter().map(|(_, min, _)| *min).collect();
                let maxes: Vec<f32> = samplers.iter().map(|(_, _, max)| *max).collect();
                let range = range_of(*first, &pointers)?;
                // SAFETY: the three arrays hold the range's length of elements and each state
                // outlives the call.
                unsafe {
                    encoder.setSamplerStates_lodMinClamps_lodMaxClamps_withRange(
                        array_head(&pointers)?,
                        array_head(&mins)?,
                        array_head(&maxes)?,
                        range,
                    );
                }
            }
            ComputeStep::ThreadgroupMemory { length, index } => {
                // SAFETY: the length is at least the bytes the kernel's `[[threadgroup(n)]]`
                // allocation reads, which the case states as one of its inputs, and the index is
                // one the kernel declares.
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(*length as usize, *index as usize);
                }
            }
            ComputeStep::UseResource {
                resource: one,
                usage,
            } => {
                let held = resource(one)?;
                encoder.useResource_usage(&held, MTLResourceUsage(*usage as usize));
            }
            ComputeStep::UseResources { resources, usage } => {
                let held: Vec<Retained<ProtocolObject<dyn MTLResource>>> =
                    resources.iter().map(resource).collect::<Answer<Vec<_>>>()?;
                let pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> = held
                    .iter()
                    .map(|one| std::ptr::NonNull::from(&**one))
                    .collect();
                // SAFETY: the array holds `count` pointers and each resource outlives the call.
                unsafe {
                    encoder.useResources_count_usage(
                        array_head(&pointers)?,
                        pointers.len(),
                        MTLResourceUsage(*usage as usize),
                    );
                }
            }
            ComputeStep::BarrierScope { scope } => {
                encoder.memoryBarrierWithScope(MTLBarrierScope(*scope as usize));
            }
            ComputeStep::BarrierResources { resources } => {
                let held: Vec<Retained<ProtocolObject<dyn MTLResource>>> =
                    resources.iter().map(resource).collect::<Answer<Vec<_>>>()?;
                let pointers: Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>> = held
                    .iter()
                    .map(|one| std::ptr::NonNull::from(&**one))
                    .collect();
                // SAFETY: as for `useResources:count:usage:`.
                unsafe {
                    encoder
                        .memoryBarrierWithResources_count(array_head(&pointers)?, pointers.len());
                }
            }
            ComputeStep::DispatchThreadgroups { groups, threads } => {
                encoder.dispatchThreadgroups_threadsPerThreadgroup(size(*groups), size(*threads));
            }
            ComputeStep::DispatchThreadgroupsIndirect {
                buffer,
                offset,
                threads,
            } => {
                let held = self.held(*buffer)?;
                // SAFETY: the buffer holds the grid's three 32-bit words at that offset, which
                // the case states as one of its inputs, and the encoder outlives the call.
                unsafe {
                    encoder
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            held,
                            *offset as usize,
                            size(*threads),
                        );
                }
            }
        }
        Ok(())
    }

    /// Encode one texture-to-buffer blit of a program into `command_buffer`.
    fn encode_read_texture(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        texture: Handle,
        into: Handle,
        offset: u64,
        row_pitch: u32,
        extent: [u32; 2],
    ) -> Answer<()> {
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            Retained::from(self.held_texture(texture)?);
        let buffer: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from(self.held(into)?);
        let bytes = row_pitch as u64 * u64::from(extent[1]);
        if u64::try_from(buffer.length()).unwrap_or(0) < offset + bytes {
            return Err(Failure::named(
                "PastTheBuffer",
                format!(
                    "{bytes} bytes at {offset} of a buffer of {}",
                    buffer.length()
                ),
            ));
        }
        let encoder = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| Failure::named("BlitCommandEncoder", "returned nil".to_owned()))?;
        // SAFETY: the buffer holds `bytes` from `offset`, checked just above, and the region is
        // level 0's whole extent of a texture the case made at that extent.
        unsafe {
            encoder
                .copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    &texture,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: extent[0] as usize,
                        height: extent[1] as usize,
                        depth: 1,
                    },
                    &buffer,
                    offset as usize,
                    row_pitch as usize,
                    bytes as usize,
                );
        }
        encoder.endEncoding();
        Ok(())
    }
}

/// A command-buffer handler block the probes keep alive: the owning half of
/// [`MTLCommandBufferHandler`], which is the raw pointer Metal's setters take.
type HandlerBlock =
    block2::RcBlock<dyn Fn(std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>;

/// A command buffer a probe waits for the scheduling of on a thread of its own.
///
/// `objc2` gives `MTLCommandBuffer` no `Send`, because Metal documents a command buffer as unsafe
/// to **encode into** from more than one thread. The probe encodes into none on that thread: it
/// calls `waitUntilScheduled`, which Metal documents as a wait any thread may make, and then
/// releases the reference. Both are sound off the thread that made it, since the retain count is
/// atomic.
struct Waited(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: as the type's documentation states: the probe never encodes into it on the waiting
// thread, and the only calls that cross threads are `waitUntilScheduled` and the release.
unsafe impl Send for Waited {}

impl Waited {
    /// `waitUntilScheduled`.
    ///
    /// A method rather than a field read at the call site, and that is load-bearing: a closure
    /// reaching `waited.0` captures the `Retained` on its own, which is not `Send`, so the whole
    /// wrapper — the thing this type's `Send` is about — would never be what crossed the thread.
    fn wait_until_scheduled(&self) {
        self.0.waitUntilScheduled();
    }
}

/// A command buffer's completion, as a bound the probe's own thread can wait on.
///
/// Metal has **no bounded wait**: `waitUntilCompleted` either returns or does not. So an event
/// probe, whose whole subject is a command buffer held back by a value that has not arrived, never
/// calls it — it adds a completed handler before the commit and waits on this channel with a bound.
/// A bound that runs out is a recorded fact ([`crate::event_orders::Status::NeverSettled`]), not a
/// hang.
struct Completion {
    done: std::sync::mpsc::Receiver<()>,
    /// The block Metal calls, kept alive for as long as the probe runs. `addCompletedHandler:`
    /// takes a pointer, and keeping our own reference means the block's lifetime does not rest on
    /// what Metal does with it.
    _block: HandlerBlock,
}

/// A completed handler on `command_buffer` that signals its channel, added before the commit.
fn completion_channel(command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Completion {
    let (sender, done) = std::sync::mpsc::channel();
    let block = block2::RcBlock::new(
        move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let _ = sender.send(());
        },
    );
    // SAFETY: the block is a valid heap block, kept alive in the `Completion` this answers for as
    // long as the probe runs, which outlives every call Metal makes of it.
    unsafe { command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&block)) };
    Completion {
        done,
        _block: block,
    }
}

/// The status `command_buffer` reads once its completion arrived, or that it never did.
fn settled(
    completion: &Completion,
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
) -> crate::event_orders::Status {
    if completion
        .done
        .recv_timeout(crate::event_orders::COMPLETION)
        .is_err()
    {
        crate::event_orders::Status::NeverSettled
    } else {
        crate::event_orders::Status::read(status_word(command_buffer.status()))
    }
}

/// The names a command buffer's handlers logged, in the order Metal called them, with each name
/// also sent so the probe can wait for one rather than guess when the log is whole.
///
/// The channel is what makes the log readable: Metal fixes no order between two handlers of one
/// command buffer that this probe could rely on, so the probe waits for the name it cares about to
/// arrive instead of assuming the handler it added first is the one that ran first.
struct HandlerLog {
    called: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    sender: std::sync::mpsc::Sender<String>,
    arrived: std::sync::mpsc::Receiver<String>,
    /// The blocks Metal calls, kept alive for as long as the probe runs.
    blocks: Vec<HandlerBlock>,
}

impl HandlerLog {
    /// An empty log.
    fn new() -> Self {
        let (sender, arrived) = std::sync::mpsc::channel();
        Self {
            called: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            sender,
            arrived,
            blocks: Vec::new(),
        }
    }

    /// Add a handler that logs itself as `name` to `command_buffer`, as the kind `add` names.
    ///
    /// `add` is the raw setter so one body serves both `addScheduledHandler:` and
    /// `addCompletedHandler:`.
    fn add(&mut self, name: &'static str, add: impl FnOnce(MTLCommandBufferHandler)) {
        let (called, sender) = (std::sync::Arc::clone(&self.called), self.sender.clone());
        let block = block2::RcBlock::new(
            move |_: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                if let Ok(mut called) = called.lock() {
                    called.push(name.to_owned());
                }
                let _ = sender.send(name.to_owned());
            },
        );
        add(block2::RcBlock::as_ptr(&block));
        self.blocks.push(block);
    }

    /// Wait until the handler named `name` has been called, or `bound` runs out.
    fn wait_for(&self, name: &str, bound: std::time::Duration) {
        let deadline = std::time::Instant::now() + bound;
        loop {
            if self
                .called
                .lock()
                .is_ok_and(|called| called.iter().any(|called| called == name))
            {
                return;
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() || self.arrived.recv_timeout(left).is_err() {
                return;
            }
        }
    }

    /// The names logged so far, in order.
    fn names(&self) -> Vec<String> {
        self.called
            .lock()
            .map(|called| called.clone())
            .unwrap_or_default()
    }
}

/// The word a command buffer's status records as, in the runtime's spelling so the two sides'
/// transcripts hold one vocabulary.
pub(crate) fn status_word(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        _ => "Unknown",
    }
}
