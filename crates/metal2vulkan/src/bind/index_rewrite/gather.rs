//! The index gather pass: the lists of a run of direct indexed draws whose programs never restart
//! ([`RewriteProgram::gathers`]: line and triangle lists, Apple's triangle fan and quads),
//! written by one dispatch for every draw of the run over one index buffer, where the index
//! rewrite pass takes three dispatches per draw (the parent module). Such a program forms its
//! primitives from its positions alone, so a draw's list is its positions' list
//! ([`RewriteProgram::sequential_list`] in the draw's order, whose start is the list of a draw of
//! fewer positions) with each position read as the fetch reads it: its index where it lies inside
//! the buffer's whole indices from the draw's offset, else 0. That is the list the rewrite pass
//! writes for the draw, index for index.
//!
//! - **The table** ([`IndexGather::table`], a storage buffer at
//!   [`IndexGatherModule::TABLE_BINDING`]): word 0 the run's draws, then from word
//!   [`IndexGatherModule::HEADER_WORDS`] one record of [`IndexGatherModule::RECORD_WORDS`] words
//!   per draw: its first workgroup, its capacity in indices, its list's first word in the output,
//!   its index offset's byte in the bound indices range, its whole indices from the offset
//!   (saturated to `u32`), and 1 for 32-bit indices.
//! - **The positions** ([`IndexGatherModule::POSITIONS_BINDING`]): the program's positions' list
//!   in the run's order, of as many indices as the run's longest list may take, which every draw
//!   of the run reads from its start.
//! - **The dispatch**: [`IndexGather::groups`] workgroups of [`IndexGatherModule::LOCAL_SIZE`]
//!   invocations, each draw's capacity over workgroups of its own from its first; a workgroup
//!   finds its draw by its first workgroup (a binary search of the records), and each invocation
//!   writes one index of its draw's list.
//! - **Bindings** (set 0, storage buffers): the table, the positions
//!   ([`IndexGather::positions_list`]), the indices ([`IndexGather::indices_range`]: the index
//!   buffer's bytes the consumer names, so one binding serves every draw of the run; any 4 bytes
//!   where it is `None`) and the output. Every read is contained (a word past the bound range
//!   reads 0), and every write lies inside the output.
//! - **Synchronization** (consumer): the indices' writes and the table's upload → `SHADER_READ`
//!   in `COMPUTE_SHADER` before the dispatch; its `SHADER_WRITE` → `INDEX_READ` in
//!   `INDEX_INPUT` before the draws.

use super::{
    boolean, contained, contained_store, finished, for_range, k, when, words_buffer, Fetch,
    RewriteOrder, RewriteProgram, Symbol, WORDS,
};
use crate::bind::index_bounds::word_range;
use crate::bind::instantiate::BufferRange;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule};
use std::sync::Arc;

impl RewriteProgram {
    /// Whether the index gather pass writes a direct draw's list (the module docs): the
    /// programs that never restart and whose positions' list for fewer positions is the start
    /// of the list for more, a line loop's not (its last segment closes on the draw's last
    /// position).
    pub fn gathers(self) -> bool {
        matches!(
            self,
            RewriteProgram::LineList
                | RewriteProgram::TriangleList
                | RewriteProgram::TriangleFan
                | RewriteProgram::Quads
        )
    }
}

/// The crate-owned gather module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexGatherModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexGatherModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The run's table ([`IndexGather::table`]), a storage buffer.
    pub const TABLE_BINDING: u32 = 0;
    /// The positions' list, a storage buffer.
    pub const POSITIONS_BINDING: u32 = 1;
    /// The Metal index buffer, a storage buffer.
    pub const INDICES_BINDING: u32 = 2;
    /// The run's lists, a storage buffer.
    pub const OUTPUT_BINDING: u32 = 3;
    /// Invocations of each workgroup, along x.
    pub const LOCAL_SIZE: u32 = 128;
    /// The table's words before its records.
    pub const HEADER_WORDS: u32 = 4;
    /// The words of one draw's record.
    pub const RECORD_WORDS: u32 = 8;
    /// The most workgroups of a run: Vulkan's least `maxComputeWorkGroupCount[0]`.
    pub const MAX_GROUPS: u32 = 65_535;
}

/// The table's binding symbol ([`IndexGatherModule::TABLE_BINDING`]).
const TABLE: ResourceSlot = WORDS;
/// The positions' binding symbol ([`IndexGatherModule::POSITIONS_BINDING`]).
const POSITIONS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The indices' binding symbol ([`IndexGatherModule::INDICES_BINDING`]).
const INDICES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The output's binding symbol ([`IndexGatherModule::OUTPUT_BINDING`]).
const OUTPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 3,
};

/// One draw of a run: Metal's index count, its index offset and the end of the bytes its fetch
/// reads (in the bound buffer's offsets), and whether its indices are 32-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherDraw {
    /// Metal's index count: the draw's positions.
    pub count: u32,
    /// The draw's index offset: a multiple of its index size.
    pub index_offset: u64,
    /// Where the draw's fetch stops reading bytes: a position from there on reads index 0.
    pub index_end: u64,
    /// 4-byte indices, else 2-byte ones.
    pub wide: bool,
}

impl GatherDraw {
    fn size(self) -> u64 {
        if self.wide {
            4
        } else {
            2
        }
    }

    /// The whole indices the draw reads inside its bytes: its count, or fewer where its fetch
    /// stops before them.
    fn inside(self) -> u32 {
        let whole = self.index_end.saturating_sub(self.index_offset) / self.size();
        whole.min(u64::from(self.count)) as u32
    }
}

/// A run of draws of one program in one order over one Metal index buffer, into one output: what
/// the consumer uploads, binds and dispatches (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexGather {
    program: RewriteProgram,
    order: RewriteOrder,
    /// The bound indices range, and the end of the bytes it binds.
    indices: Option<(BufferRange, u64)>,
    output_words: u32,
    positions: u32,
    /// Each draw, its first workgroup and its list's first word.
    draws: Vec<(GatherDraw, u32, u32)>,
    groups: u32,
    written: u32,
}

impl IndexGather {
    /// An empty run of `program` in `order` over an index buffer whose bytes from `index_start`
    /// to `index_end` it binds (its `VkBuffer` holds the bytes to `index_end` rounded up to 4),
    /// at an offset aligned to `alignment` (`minStorageBufferOffsetAlignment`) over at most
    /// `max_range` bytes (`maxStorageBufferRange`), into an output of `output_words` words, of
    /// draws whose lists take at most `positions` indices each. `None` for a program that does
    /// not gather ([`RewriteProgram::gathers`]), and where the bytes are past `max_range`.
    pub fn new(
        (program, order): (RewriteProgram, RewriteOrder),
        (index_start, index_end): (u64, u64),
        (alignment, max_range): (u64, u32),
        (output_words, positions): (u32, u32),
    ) -> Option<IndexGather> {
        if !program.gathers() {
            return None;
        }
        let indices = match index_end > index_start {
            true => {
                let (range, _) = word_range(index_start, index_end, alignment, max_range)?;
                Some((range, index_end))
            }
            false => None,
        };
        Some(IndexGather {
            program,
            order,
            indices,
            output_words,
            positions,
            draws: Vec::new(),
            groups: 0,
            written: 0,
        })
    }

    /// The indices of `draw`'s list ([`RewriteProgram::capacity`] of its count), where they fit
    /// a `u32`.
    pub fn capacity(&self, draw: GatherDraw) -> Option<u32> {
        u32::try_from(self.program.capacity(u64::from(draw.count))).ok()
    }

    /// Add `draw` to the run: its list's first word in the output. `None` where the run cannot
    /// take it, and is as before: a draw of no primitive, an offset not a multiple of the index
    /// size, indices read outside the bound bytes, a list longer than the run's positions or past
    /// the output, or workgroups past [`IndexGatherModule::MAX_GROUPS`].
    pub fn push(&mut self, draw: GatherDraw) -> Option<u32> {
        let capacity = self.capacity(draw)?;
        if capacity == 0
            || capacity > self.positions
            || !draw.index_offset.is_multiple_of(draw.size())
        {
            return None;
        }
        if draw.inside() > 0 {
            let (range, end) = self.indices?;
            if draw.index_offset < range.offset || draw.index_end > end {
                return None;
            }
        }
        let first = self.written;
        let written = first.checked_add(capacity)?;
        let groups = self
            .groups
            .checked_add(capacity.div_ceil(IndexGatherModule::LOCAL_SIZE))?;
        if written > self.output_words || groups > IndexGatherModule::MAX_GROUPS {
            return None;
        }
        self.draws.push((draw, self.groups, first));
        self.written = written;
        self.groups = groups;
        Some(first)
    }

    /// The run's draws.
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// Whether the run has no draw.
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// The workgroups of the run's dispatch.
    pub fn groups(&self) -> u32 {
        self.groups
    }

    /// The output's words the run's lists take.
    pub fn written(&self) -> u32 {
        self.written
    }

    /// The range of the Metal index buffer to bind at [`IndexGatherModule::INDICES_BINDING`]:
    /// the bytes [`new`](Self::new) names, from their start aligned down, to a whole word; `None`
    /// where they are none.
    pub fn indices_range(&self) -> Option<BufferRange> {
        self.indices.map(|(range, _)| range)
    }

    /// The positions' list to bind at [`IndexGatherModule::POSITIONS_BINDING`]: the program's
    /// in the run's order ([`RewriteProgram::sequential_list`]), of the run's positions.
    pub fn positions_list(&self) -> Vec<u32> {
        let want = u64::from(self.positions);
        // Twice the positions and four more form at least as many indices for every program
        // that gathers.
        let count = u32::try_from(want * 2 + 4).unwrap_or(u32::MAX);
        let mut list = self.program.sequential_list(count, self.order);
        list.truncate(self.positions as usize);
        list
    }

    /// The table to upload: the draws, then each draw's record (the module docs).
    pub fn table(&self) -> Vec<u32> {
        let header = IndexGatherModule::HEADER_WORDS as usize;
        let mut table = vec![0; header];
        table[0] = self.draws.len() as u32;
        let start = self.indices.map_or(0, |(range, _)| range.offset);
        for &(draw, first_group, first_word) in &self.draws {
            let inside = draw.inside();
            // A draw with no index inside reads none: its byte is never read.
            let byte = match inside {
                0 => 0,
                _ => draw.index_offset - start,
            };
            let mut record = [0; IndexGatherModule::RECORD_WORDS as usize];
            record[0] = first_group;
            record[1] = self.capacity(draw).unwrap_or(0);
            record[2] = first_word;
            record[3] = byte as u32;
            record[4] = inside;
            record[5] = u32::from(draw.wide);
            table.extend(record);
        }
        table
    }

    /// The table's bytes ([`table`](Self::table)).
    pub fn table_bytes(&self) -> Vec<u8> {
        self.table().iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// The gather dispatch as a typed module, before serialization: the module
/// [`index_gather_module`] serializes, which the CPU interpreter runs.
pub(crate) fn gather_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let table = words_buffer(&mut b, TABLE, u32_)?;
    let positions = words_buffer(&mut b, POSITIONS, u32_)?;
    let indices = words_buffer(&mut b, INDICES, u32_)?;
    let output = words_buffer(&mut b, OUTPUT, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let workgroup = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let table = f.module().global_value(table)?;
    let positions = f.module().global_value(positions)?;
    let indices = f.module().global_value(indices)?;
    let output = f.module().global_value(output)?;
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let workgroup = f.module().global_value(workgroup)?;
    let workgroup = f.load(workgroup, MemAccess::default())?;
    let group = f.composite_extract(workgroup, &[0])?;
    let zero = k(&mut f, 0)?;
    let one = k(&mut f, 1)?;
    let header = k(&mut f, IndexGatherModule::HEADER_WORDS)?;
    let per = k(&mut f, IndexGatherModule::RECORD_WORDS)?;
    // The draw: the last record whose first workgroup is at or before this one's, the first's
    // being 0. Each step halves `[lo, hi)`; 32 steps end every search a `u32` counts.
    let draws = contained(&mut f, table, zero)?;
    let lo_var = f.local_var(u32_, None)?;
    let lo = f.var_ptr(lo_var)?;
    f.store(lo, zero, MemAccess::default())?;
    let hi_var = f.local_var(u32_, None)?;
    let hi = f.var_ptr(hi_var)?;
    f.store(hi, draws, MemAccess::default())?;
    let steps = k(&mut f, 32)?;
    for_range(&mut f, u32_, (zero, steps, one), move |f, _| {
        let l = f.load(lo, MemAccess::default())?;
        let h = f.load(hi, MemAccess::default())?;
        let span = f.binary(BinOp::ISub, h, l)?;
        let one = k(f, 1)?;
        let wide = f.binary(BinOp::UGreaterThan, span, one)?;
        when(f, wide, move |f| {
            let sum = f.binary(BinOp::IAdd, l, h)?;
            let one = k(f, 1)?;
            let mid = f.binary(BinOp::ShiftRightLogical, sum, one)?;
            let header = k(f, IndexGatherModule::HEADER_WORDS)?;
            let per = k(f, IndexGatherModule::RECORD_WORDS)?;
            let at = f.binary(BinOp::IMul, mid, per)?;
            let at = f.binary(BinOp::IAdd, header, at)?;
            let first = contained(f, table, at)?;
            let before = f.binary(BinOp::ULessThanEqual, first, group)?;
            let l2 = f.select(before, mid, l)?;
            let h2 = f.select(before, h, mid)?;
            f.store(lo, l2, MemAccess::default())?;
            f.store(hi, h2, MemAccess::default())
        })
    })?;
    let found = f.load(lo, MemAccess::default())?;
    let at = f.binary(BinOp::IMul, found, per)?;
    let record = f.binary(BinOp::IAdd, header, at)?;
    let word = |f: &mut FunctionBuilder<'_>, i: u32| -> BResult<_> {
        let i = k(f, i)?;
        let at = f.binary(BinOp::IAdd, record, i)?;
        contained(f, table, at)
    };
    let first_group = word(&mut f, 0)?;
    let capacity = word(&mut f, 1)?;
    let first_word = word(&mut f, 2)?;
    let first_byte = word(&mut f, 3)?;
    let inside = word(&mut f, 4)?;
    let wide = word(&mut f, 5)?;
    let wide = f.binary(BinOp::INotEqual, wide, zero)?;
    let local = k(&mut f, IndexGatherModule::LOCAL_SIZE)?;
    let groups_in = f.binary(BinOp::ISub, group, first_group)?;
    let o = f.binary(BinOp::IMul, groups_in, local)?;
    let o = f.binary(BinOp::IAdd, o, lin)?;
    let has = f.binary(BinOp::ULessThan, zero, draws)?;
    let within = f.binary(BinOp::ULessThan, o, capacity)?;
    let writes = f.binary(BinOp::LogicalAnd, has, within)?;
    let sequential = boolean(&mut f, false)?;
    let fetch = Fetch {
        indices,
        first_byte,
        inside,
        start: zero,
        wide,
        sequential,
    };
    when(&mut f, writes, move |f| {
        let position = contained(f, positions, o)?;
        let value = fetch.at(f, position)?;
        let at = f.binary(BinOp::IAdd, first_word, o)?;
        contained_store(f, output, at, value)
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IndexGatherModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexGatherModule::SET),
        Symbol::Binding(s) if *s == TABLE => Some(IndexGatherModule::TABLE_BINDING),
        Symbol::Binding(s) if *s == POSITIONS => Some(IndexGatherModule::POSITIONS_BINDING),
        Symbol::Binding(s) if *s == INDICES => Some(IndexGatherModule::INDICES_BINDING),
        Symbol::Binding(s) if *s == OUTPUT => Some(IndexGatherModule::OUTPUT_BINDING),
        _ => None,
    }
}

/// The index gather module for `device`: built with the typed builder, verified, serialized, and
/// numbered with the fixed bindings of [`IndexGatherModule`].
pub fn index_gather_module(device: &Device) -> Result<IndexGatherModule, Error> {
    let (spirv, requirements) = finished(
        device,
        gather_typed_module()?,
        binding_of,
        "the index gather",
    )?;
    Ok(IndexGatherModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
