//! The per-call cost table (`--only percall`): one group a call, each call timed alone on both
//! sides, with Metal's cost as the budget.
//!
//! A group's window prepares the state its call needs, untimed; makes that one call `inner`
//! times; then drains the state, committing and waiting where the calls encoded work. Three
//! warm windows run first, and every pipeline state, library and texture the groups bind is made
//! before any of them, so no making falls inside a timed window (the `variant_pipelines_made`
//! count a group reports is the proof).
//!
//! Per call a group reports
//!
//! - `<call>_self`: the calling thread's wall time, the cost the caller pays inline;
//! - `<call>_blocked`: how much of that the calling thread was off a CPU, so it waited;
//! - `<call>_others`: our own other threads' CPU time while the calls ran;
//! - `<call>_drain`: the calling thread's wall time in the drain after the calls, where the
//!   group has one (its command buffer's commit and wait): what the caller pays later;
//! - `<call>_drain_blocked`: how much of that drain the calling thread was off a CPU, so a
//!   drain that costs CPU instead was spent spinning;
//! - `<call>_drain_others`: our other threads' CPU time in that drain;
//! - `<call>_allocs`: the heap allocations a call made on the calling thread;
//! - `<call>_rt_<count>`: the runtime's own counts of what the calls made, where they moved
//!   ([`COUNTERS`]); none natively, so these rows have no ratio.
//!
//! The wall times are a median over the windows, and the rest the group's totals over its
//! calls, because the thread and process CPU clocks answer in microseconds.

use crate::gpu::{Format, Gpu};
use crate::workloads::{emit, host_pages, process_cpu, thread_cpu, Scene, Sizes, TARGET_SIDE};
use std::time::{Duration, Instant};

/// The windows a group runs before the measured ones.
const WARM: usize = 3;

/// The runtime's counts a group reports per call where they moved: what its calls made the
/// driver, and our own threads, do ([`Gpu::work_counts`]; none natively).
const COUNTERS: [&str; 16] = [
    "queue_submissions",
    "pipeline_barriers",
    "draws",
    "dispatches",
    "render_passes_begun",
    "requirement_images_made",
    "images_own_memory",
    "arena_pools_made",
    "command_pools_made",
    "wakes_handler_threads",
    "scheduler_passes",
    "variant_pipelines_made",
    "host_pointer_imports_made",
    "completed_waits",
    "device_memory_bytes",
    "cleared_bytes",
];

/// What one window measured: the calls' wall time on the calling thread, that thread's CPU time
/// and the whole process's while they ran, then the same three of the drain after them.
#[derive(Clone, Copy, Default)]
struct Sample {
    wall: Duration,
    cpu: Duration,
    process: Duration,
    drain_wall: Duration,
    drain_cpu: Duration,
    drain_process: Duration,
}

/// A group's windows, summed.
#[derive(Default)]
struct Totals {
    /// Each window's wall time a call, in ns: the call's own cost.
    own: Vec<f64>,
    /// Each window's drain, in ns a call.
    drain: Vec<f64>,
    sums: Sample,
}

impl Totals {
    fn add(&mut self, inner: usize, s: Sample) {
        let each = |d: Duration| d.as_nanos() as f64 / inner as f64;
        self.own.push(each(s.wall));
        self.drain.push(each(s.drain_wall));
        let sums = &mut self.sums;
        sums.wall += s.wall;
        sums.cpu += s.cpu;
        sums.process += s.process;
        sums.drain_wall += s.drain_wall;
        sums.drain_cpu += s.drain_cpu;
        sums.drain_process += s.drain_process;
    }
}

/// Time one call group: `windows` measured windows, each preparing the call's state, making the
/// call `inner` times, then draining it. Every window runs inside an autorelease pool, as a
/// Metal program's frame does, so nothing of one window is released inside another's timing.
fn group<G: Gpu, S>(
    g: &G,
    name: &str,
    (inner, windows): (usize, usize),
    mut prepare: impl FnMut() -> S,
    mut call: impl FnMut(&mut S, usize),
    mut drain: impl FnMut(S),
) {
    assert!(inner > 0 && windows > 0, "{name}: no calls");
    let mut totals = Totals::default();
    let mut before = vec![];
    for window in 0..WARM + windows {
        if window == WARM {
            before = g.work_counts();
        }
        let sample = g.pool(|| {
            let mut state = prepare();
            let (cpu0, process0) = (thread_cpu(), process_cpu());
            let start = Instant::now();
            for i in 0..inner {
                call(&mut state, i);
            }
            let called = Instant::now();
            let (cpu1, process1) = (thread_cpu(), process_cpu());
            let drained_from = Instant::now();
            drain(state);
            let drained = Instant::now();
            let (cpu2, process2) = (thread_cpu(), process_cpu());
            Sample {
                wall: called - start,
                cpu: cpu1 - cpu0,
                process: process1 - process0,
                drain_wall: drained - drained_from,
                drain_cpu: cpu2 - cpu1,
                drain_process: process2 - process1,
            }
        });
        if window >= WARM {
            totals.add(inner, sample);
        }
    }
    let after = g.work_counts();
    let calls = (inner * windows) as f64;
    let each = |d: Duration| d.as_nanos() as f64 / calls;
    let sums = &totals.sums;
    let metric = |what: &str| format!("{name}_{what}");
    emit("percall", &metric("self"), "ns/call", &mut totals.own);
    emit(
        "percall",
        &metric("blocked"),
        "ns/call",
        &mut [(each(sums.wall) - each(sums.cpu)).max(0.0)],
    );
    emit(
        "percall",
        &metric("others"),
        "ns/call",
        &mut [(each(sums.process) - each(sums.cpu)).max(0.0)],
    );
    emit("percall", &metric("drain"), "ns/call", &mut totals.drain);
    emit(
        "percall",
        &metric("drain_blocked"),
        "ns/call",
        &mut [(each(sums.drain_wall) - each(sums.drain_cpu)).max(0.0)],
    );
    emit(
        "percall",
        &metric("drain_others"),
        "ns/call",
        &mut [(each(sums.drain_process) - each(sums.drain_cpu)).max(0.0)],
    );
    // One window more, this thread's allocations counted.
    let made = g.pool(|| {
        let mut state = prepare();
        let made = crate::allocs::counting(|| {
            let before = crate::allocs::this_thread();
            for i in 0..inner {
                call(&mut state, i);
            }
            crate::allocs::this_thread() - before
        });
        drain(state);
        made
    });
    emit(
        "percall",
        &metric("allocs"),
        "allocs/call",
        &mut [made as f64 / inner as f64],
    );
    // What the measured windows' calls made the runtime do, per call.
    for (counted, from) in &before {
        if !COUNTERS.contains(counted) {
            continue;
        }
        let now = after
            .iter()
            .find(|(name, _)| name == counted)
            .map_or(0.0, |(_, count)| *count);
        assert!(
            *counted != "variant_pipelines_made" || now <= *from,
            "{name}: {} pipeline states were made inside its timed windows",
            now - from
        );
        if now > *from {
            emit(
                "percall",
                &metric(&format!("rt_{counted}")),
                "count/call",
                &mut [(now - from) / calls],
            );
        }
    }
}

/// How many bytes of host pages the no-copy buffers of one window lie over, and how many such
/// buffers a window makes: one megabyte each, the same pages every window, as a guest's buffers
/// lie in its memory.
const NO_COPY_EACH: usize = 1 << 20;
const NO_COPY_BUFFERS: usize = 8;

/// The buffers a residency group makes resident, 256 bytes each.
const RESIDENT: usize = 256;

/// The side of the textures the creation and upload groups make.
const SMALL_SIDE: usize = 256;

/// The bytes a small texture's level holds, as `BGRA8Unorm` of unpadded rows.
const SMALL_BYTES: usize = SMALL_SIDE * SMALL_SIDE * 4;

/// The textures a placement heap holds in the heap-texture and `useHeap` groups.
const HEAP_TEXTURES: usize = 16;

/// What the groups bind, draw and copy: made once, before any window.
struct Objects<G: Gpu> {
    /// A render pass of the scene's target, cleared and stored, made once for many encoders.
    pass: G::Pass,
    /// `calls_vertex` with the texture-reading and the buffer-reading fragment function.
    textured: G::RenderPipeline,
    buffered: G::RenderPipeline,
    /// `quad_vertex` + `textured_fragment` into the depth pass's attachments.
    depth_state: G::RenderPipeline,
    depth: G::Texture,
    /// A depth compare with writes, and one without.
    depth_stencils: [G::DepthStencil; 2],
    /// The vertex buffers `calls_vertex` reads, and the fragment buffer `calls_buffer_fragment`
    /// does.
    positions: G::Buffer,
    colours: G::Buffer,
    uniforms: G::Buffer,
    /// Sixteen 16-bit indices of a triangle.
    indices: G::Buffer,
    /// A compute encoder's bound buffers, and a blit's source and destination.
    input: G::Buffer,
    output: G::Buffer,
    from: G::Buffer,
    to: G::Buffer,
    /// A small texture's bytes, for the upload and `replaceRegion` groups, and a texture of that
    /// size to upload into and read back.
    texels: Vec<u8>,
    small: G::Buffer,
    small_texture: G::Texture,
    /// The buffers the residency groups make resident.
    resident: Vec<G::Buffer>,
    /// A placement heap of `Shared` storage, whose hazards nothing tracks: what `useHeap` is
    /// for, and where the heap-texture group places its textures.
    heap: G::Heap,
    /// A shared event, a fence, and the pages the no-copy buffers lie over.
    event: G::SharedEvent,
    fence: G::Fence,
    pages: *mut u8,
    /// A command buffer of one fill, committed and completed, whose status, times and wait the
    /// read groups take.
    done: G::CommandBuffer,
}

impl<G: Gpu> Objects<G> {
    fn new(g: &G, s: &Scene<G>, library: &G::Library) -> Objects<G> {
        let state =
            |fragment| g.render_pipeline(library, "calls_vertex", fragment, Format::Bgra8, false);
        let floats =
            |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        // A strip of four corners of a small square, then nothing, as the calls workload's.
        let positions: Vec<f32> = (0..256)
            .flat_map(|v| match v {
                0..4 => [
                    -0.1 + 0.2 * (v & 1) as f32,
                    -0.1 + 0.2 * (v >> 1) as f32,
                    0.0,
                    1.0,
                ],
                _ => [0.0; 4],
            })
            .collect();
        let texels: Vec<u8> = (0..SMALL_BYTES).map(|i| (i % 251) as u8).collect();
        let done = g.command_buffer();
        g.fill(&done, &s.fill, 256);
        g.commit(&done);
        g.wait(&done);
        Objects {
            pass: g.pass(&s.target),
            textured: state("calls_texture_fragment"),
            buffered: state("calls_buffer_fragment"),
            depth_state: g.depth_pipeline(library, "quad_vertex", "textured_fragment", false),
            depth: g.texture(Format::Depth32Float, TARGET_SIDE, TARGET_SIDE, true),
            depth_stencils: [g.depth_stencil(true), g.depth_stencil(false)],
            positions: g.buffer_with(&floats(&positions)),
            colours: g.buffer_with(&floats(&[0.5f32, 0.25, 0.75, 1.0].repeat(256))),
            uniforms: g.buffer_with(&floats(&[0.25f32, 0.5, 0.75, 1.0].repeat(4096))),
            indices: g.buffer_with(&(0..16u16).flat_map(u16::to_le_bytes).collect::<Vec<u8>>()),
            input: g.buffer_with(&vec![1u8; 1 << 16]),
            output: g.buffer(1 << 16),
            from: g.buffer_with(&vec![0x3cu8; 1 << 18]),
            to: g.buffer(1 << 18),
            small: g.buffer_with(&texels),
            small_texture: g.texture(Format::Bgra8, SMALL_SIDE, SMALL_SIDE, false),
            texels,
            resident: (0..RESIDENT).map(|_| g.buffer(256)).collect(),
            heap: g.heap(HEAP_TEXTURES * SMALL_BYTES * 2),
            event: g.shared_event(),
            fence: g.fence(),
            // SAFETY: the pages outlive every buffer made over them; nothing unmaps them.
            pages: unsafe { host_pages(NO_COPY_BUFFERS * NO_COPY_EACH) },
            done,
        }
    }
}

/// The per-call table: every group, in the order a frame makes its calls.
pub fn percall<G: Gpu>(g: &G, s: &Scene<G>, library: &G::Library, sizes: &Sizes) {
    let o = Objects::new(g, s, library);
    let w = sizes.timed;
    // A group whose window makes one call, because a second call of it would not pay what the
    // first did, runs more windows: its samples are its windows, and the host clock this Mac
    // counts steps in 41.67 ns, so a call of a few tens of nanoseconds needs many of them.
    let lone = w * 6;
    let made_before = g.work_counts();

    // The command buffer: made, committed empty, committed with one encoder's work, and the
    // status, times and completed wait a frame's end reads.
    group(
        g,
        "command_buffer",
        (32, w),
        || Vec::with_capacity(32),
        |held: &mut Vec<G::CommandBuffer>, _| held.push(g.command_buffer()),
        drop,
    );
    group(
        g,
        "commit_empty",
        (16, w),
        || (0..16).map(|_| g.command_buffer()).collect::<Vec<_>>(),
        |held: &mut Vec<G::CommandBuffer>, i| g.commit(&held[i]),
        |held| held.iter().for_each(|cb| g.wait(cb)),
    );
    group(
        g,
        "commit_one_dispatch",
        (16, w),
        || {
            (0..16)
                .map(|_| {
                    let cb = g.command_buffer();
                    let mut e = g.compute_encoder(&cb);
                    g.set_compute_pipeline(&mut e, &s.tiny[0]);
                    g.set_buffer(&mut e, &o.output, 0, 0);
                    g.set_buffer(&mut e, &o.input, 0, 1);
                    g.dispatch(&mut e, 1, 64);
                    g.end_compute(e);
                    cb
                })
                .collect::<Vec<_>>()
        },
        |held: &mut Vec<G::CommandBuffer>, i| g.commit(&held[i]),
        |held| held.iter().for_each(|cb| g.wait(cb)),
    );
    group(
        g,
        "on_completed",
        (16, w),
        || (0..16).map(|_| g.command_buffer()).collect::<Vec<_>>(),
        |held: &mut Vec<G::CommandBuffer>, i| g.on_completed(&held[i], Box::new(|| {})),
        |held| {
            held.iter().for_each(|cb| g.commit(cb));
            held.iter().for_each(|cb| g.wait(cb));
        },
    );
    assert_eq!(g.status(&o.done), 4, "the read groups' command buffer");
    group(
        g,
        "status_completed",
        (256, w),
        || (),
        |_, _| assert_eq!(std::hint::black_box(g.status(&o.done)), 4),
        drop,
    );
    group(
        g,
        "times_completed",
        (256, w),
        || (),
        |_, _| {
            std::hint::black_box(g.times(&o.done));
        },
        drop,
    );
    group(
        g,
        "wait_completed",
        (256, w),
        || (),
        |_, _| g.wait_only(&o.done),
        drop,
    );

    // The encoders: opened and ended. A window opens or ends one encoder, so neither call pays
    // for the other, and the command buffer the window built is committed and waited in its
    // drain, so the work our threads do for it lands in the drain's columns.
    group(
        g,
        "render_encoder_pass",
        (1, lone),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| {
            let e = g.pass_encoder(cb, &o.pass);
            g.end_render(e);
        },
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    for (name, clear) in [
        ("render_encoder_clear", true),
        ("render_encoder_load", false),
    ] {
        group(
            g,
            name,
            (1, lone),
            || g.command_buffer(),
            |cb: &mut G::CommandBuffer, _| {
                let e = g.render_encoder(cb, &s.target, clear);
                g.end_render(e);
            },
            |cb| {
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    }
    group(
        g,
        "end_render",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let e = g.pass_encoder(&cb, &o.pass);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::RenderEncoder>), _| {
            g.end_render(state.1.take().expect("the window's encoder"))
        },
        |(cb, _)| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "compute_encoder",
        (1, lone),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| {
            let e = g.compute_encoder(cb);
            g.end_compute(e);
        },
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "end_compute",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let mut e = g.compute_encoder(&cb);
            g.set_compute_pipeline(&mut e, &s.tiny[0]);
            g.set_buffer(&mut e, &o.output, 0, 0);
            g.set_buffer(&mut e, &o.input, 0, 1);
            g.dispatch(&mut e, 1, 64);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::ComputeEncoder>), _| {
            g.end_compute(state.1.take().expect("the window's encoder"))
        },
        |(cb, _)| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "blit_encoder",
        (1, lone),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| {
            let e = g.blit_encoder(cb);
            g.end_blit(e);
        },
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "end_blit",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let mut e = g.blit_encoder(&cb);
            g.copy_buffer(&mut e, (&o.from, 0), (&o.to, 0), 1024);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::BlitEncoder>), _| {
            g.end_blit(state.1.take().expect("the window's encoder"))
        },
        |(cb, _)| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The state a draw binds. Each window's encoder is committed and waited, so the binds' later
    // cost shows in the drain.
    let render = |name: &str, binds: usize, bind: &dyn Fn(&mut G::RenderEncoder, usize)| {
        group(
            g,
            name,
            (binds, w),
            || {
                let cb = g.command_buffer();
                let mut e = g.pass_encoder(&cb, &o.pass);
                g.set_render_pipeline(&mut e, &o.textured);
                g.set_vertex_buffers(&mut e, &[&o.positions, &o.colours], &[0, 0], 0);
                g.set_fragment_texture(&mut e, &s.textures[0], 0);
                g.set_fragment_sampler(&mut e, &s.sampler, 0);
                (cb, Some(e))
            },
            |state: &mut (G::CommandBuffer, Option<G::RenderEncoder>), i| {
                bind(state.1.as_mut().expect("the window's encoder"), i)
            },
            |(cb, e)| {
                g.end_render(e.expect("the window's encoder"));
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    };
    render("set_render_pipeline", 256, &|e, i| {
        g.set_render_pipeline(e, [&o.textured, &o.buffered][i % 2])
    });
    render("set_vertex_buffer", 256, &|e, i| {
        g.set_vertex_buffer(e, &o.positions, (i % 64) * 16, 0)
    });
    render("set_vertex_buffers_two", 256, &|e, i| {
        g.set_vertex_buffers(e, &[&o.positions, &o.colours], &[0, (i % 64) * 16], 0)
    });
    const SIXTEEN: [u8; 16] = [0x3c; 16];
    render("set_vertex_bytes_16", 256, &|e, _| {
        g.set_vertex_bytes(e, &SIXTEEN, 2)
    });
    render("set_fragment_buffer", 256, &|e, i| {
        g.set_fragment_buffer(e, &o.uniforms, (i % 256) * 256, 0)
    });
    render("set_fragment_texture", 256, &|e, i| {
        g.set_fragment_texture(e, &s.textures[i % s.textures.len()], 0)
    });
    render("set_fragment_sampler", 256, &|e, _| {
        g.set_fragment_sampler(e, &s.sampler, 0)
    });
    render("set_cull_mode", 256, &|e, i| g.set_cull_back(e, i % 2 == 0));
    let resident: Vec<&G::Buffer> = o.resident.iter().collect();
    render("use_resources_read_256_drawn", 1, &|e, _| {
        g.use_buffers_drawn(e, &resident)
    });
    render("use_resources_read_1_drawn", 1, &|e, _| {
        g.use_buffers_drawn(e, &[&o.resident[0]])
    });

    // The draws, each of a strip of four vertices, and the state they draw under set once.
    render("draw_strip", 512, &|e, _| g.draw_strip(e, 4));
    render("draw_triangles", 512, &|e, _| g.draw_triangles(e, 0, 3));
    render("draw_indexed", 512, &|e, _| {
        g.draw_indexed(e, 3, &o.indices)
    });

    // A depth pass: the depth-stencil state, which no other group binds.
    group(
        g,
        "set_depth_stencil",
        (256, w),
        || {
            let cb = g.command_buffer();
            let mut e = g.depth_render_encoder(&cb, &s.target, &o.depth);
            g.set_render_pipeline(&mut e, &o.depth_state);
            g.set_vertex_buffer(&mut e, &s.rects[0], 0, 0);
            g.set_fragment_texture(&mut e, &s.textures[0], 0);
            g.set_fragment_sampler(&mut e, &s.sampler, 0);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::RenderEncoder>), i| {
            g.set_depth_stencil(
                state.1.as_mut().expect("the window's encoder"),
                &o.depth_stencils[i % 2],
            )
        },
        |(cb, e)| {
            g.end_render(e.expect("the window's encoder"));
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The compute encoder's calls.
    let compute = |name: &str, calls: usize, call: &dyn Fn(&mut G::ComputeEncoder, usize)| {
        group(
            g,
            name,
            (calls, w),
            || {
                let cb = g.command_buffer();
                let mut e = g.compute_encoder(&cb);
                g.set_compute_pipeline(&mut e, &s.tiny[0]);
                g.set_buffer(&mut e, &o.output, 0, 0);
                g.set_buffer(&mut e, &o.input, 0, 1);
                (cb, Some(e))
            },
            |state: &mut (G::CommandBuffer, Option<G::ComputeEncoder>), i| {
                call(state.1.as_mut().expect("the window's encoder"), i)
            },
            |(cb, e)| {
                g.end_compute(e.expect("the window's encoder"));
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    };
    compute("set_compute_pipeline", 256, &|e, i| {
        g.set_compute_pipeline(e, &s.tiny[i % 2])
    });
    compute("set_buffer", 256, &|e, i| {
        g.set_buffer(e, &o.output, (i % 64) * 256, 0)
    });
    compute("set_texture", 256, &|e, i| {
        g.set_texture(e, &s.textures[i % s.textures.len()], 0)
    });
    compute("dispatch", 256, &|e, _| g.dispatch(e, 1, 64));
    compute("use_resources_read_256", 1, &|e, _| {
        g.use_buffers_read(e, &resident)
    });
    compute("use_resources_read_1", 1, &|e, _| {
        g.use_buffers_read(e, &[&o.resident[0]])
    });
    compute("use_resource_read_write", 1, &|e, _| {
        g.use_buffer(e, &o.resident[0])
    });
    compute("use_heap", 1, &|e, _| g.use_heap(e, &o.heap));
    compute("update_fence", 1, &|e, _| g.update_fence(e, &o.fence));
    // The fence this encoder waits for was updated by the encoder before it in the same command
    // buffer, as Metal's fences are used, so nothing waits for an update that never comes.
    group(
        g,
        "wait_for_fence",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let mut first = g.compute_encoder(&cb);
            g.set_compute_pipeline(&mut first, &s.tiny[0]);
            g.set_buffer(&mut first, &o.output, 0, 0);
            g.set_buffer(&mut first, &o.input, 0, 1);
            g.dispatch(&mut first, 1, 64);
            g.update_fence(&mut first, &o.fence);
            g.end_compute(first);
            let mut e = g.compute_encoder(&cb);
            g.set_compute_pipeline(&mut e, &s.tiny[0]);
            g.set_buffer(&mut e, &o.output, 0, 0);
            g.set_buffer(&mut e, &o.input, 0, 1);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::ComputeEncoder>), _| {
            g.wait_for_fence(state.1.as_mut().expect("the window's encoder"), &o.fence)
        },
        |(cb, e)| {
            let mut e = e.expect("the window's encoder");
            g.dispatch(&mut e, 1, 64);
            g.end_compute(e);
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The blit encoder's calls, and the whole blit encoder one fill takes.
    let blit = |name: &str, calls: usize, call: &dyn Fn(&mut G::BlitEncoder, usize)| {
        group(
            g,
            name,
            (calls, w),
            || {
                let cb = g.command_buffer();
                let e = g.blit_encoder(&cb);
                (cb, Some(e))
            },
            |state: &mut (G::CommandBuffer, Option<G::BlitEncoder>), i| {
                call(state.1.as_mut().expect("the window's encoder"), i)
            },
            |(cb, e)| {
                g.end_blit(e.expect("the window's encoder"));
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    };
    blit("copy_buffer_1k", 256, &|e, i| {
        let at = (i % 256) * 1024;
        g.copy_buffer(e, (&o.from, at), (&o.to, at), 1024)
    });
    blit("fill_buffer_256", 256, &|e, i| {
        g.fill_buffer_in(e, &o.to, (i % 256) * 1024, 256)
    });
    blit("copy_buffer_to_texture", 16, &|e, _| {
        g.copy_to_texture(
            e,
            &o.small,
            &o.small_texture,
            (SMALL_SIDE, SMALL_SIDE),
            SMALL_SIDE * 4,
        )
    });
    group(
        g,
        "fill_in_own_encoder",
        (16, w),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| g.fill(cb, &o.to, 256),
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The objects a frame makes and drops: each window's are released before the next window's,
    // so no group runs in memory the one before it filled.
    group(
        g,
        "buffer_4k",
        (64, w),
        || Vec::with_capacity(64),
        |held: &mut Vec<G::Buffer>, _| held.push(g.buffer(4096)),
        drop,
    );
    group(
        g,
        "buffer_1m",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::Buffer>, _| held.push(g.buffer(1 << 20)),
        drop,
    );
    group(
        g,
        "buffer_with_bytes_4k",
        (64, w),
        || Vec::with_capacity(64),
        |held: &mut Vec<G::Buffer>, _| held.push(g.buffer_with(&o.texels[..4096])),
        drop,
    );
    group(
        g,
        "buffer_no_copy_1m",
        (NO_COPY_BUFFERS, w),
        || Vec::with_capacity(NO_COPY_BUFFERS),
        |held: &mut Vec<G::Buffer>, i| {
            // SAFETY: the pages are page-aligned read-write memory of this many bytes, and they
            // outlive every buffer made over them (`Objects::pages`).
            held.push(unsafe { g.buffer_no_copy(o.pages.add(i * NO_COPY_EACH), NO_COPY_EACH) })
        },
        drop,
    );
    group(
        g,
        "texture_256_shared",
        (32, w),
        || Vec::with_capacity(32),
        |held: &mut Vec<G::Texture>, _| {
            held.push(g.texture(Format::Bgra8, SMALL_SIDE, SMALL_SIDE, false))
        },
        drop,
    );
    group(
        g,
        "texture_1024_target",
        (8, w),
        || Vec::with_capacity(8),
        |held: &mut Vec<G::Texture>, _| held.push(g.texture(Format::Bgra8, 1024, 1024, true)),
        drop,
    );
    group(
        g,
        "heap_texture_256",
        (HEAP_TEXTURES, w),
        || Vec::with_capacity(HEAP_TEXTURES),
        |held: &mut Vec<G::Texture>, i| {
            held.push(g.heap_texture(&o.heap, SMALL_SIDE, SMALL_SIDE, i * SMALL_BYTES))
        },
        drop,
    );
    group(
        g,
        "buffer_texture_256",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::Texture>, _| {
            held.push(g.buffer_texture(&o.small, (SMALL_SIDE, SMALL_SIDE), 0, SMALL_SIDE * 4))
        },
        drop,
    );
    group(
        g,
        "sampler",
        (32, w),
        || Vec::with_capacity(32),
        |held: &mut Vec<G::Sampler>, i| held.push(g.sampler(i % 2 == 0)),
        drop,
    );
    group(
        g,
        "shared_event",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::SharedEvent>, _| held.push(g.shared_event()),
        drop,
    );
    group(
        g,
        "fence",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::Fence>, _| held.push(g.fence()),
        drop,
    );
    group(
        g,
        "event_value",
        (256, w),
        || (),
        |_, _| {
            std::hint::black_box(g.event_value(&o.event));
        },
        drop,
    );
    let next_value = std::cell::Cell::new(1u64);
    group(
        g,
        "set_event_value",
        (256, w),
        || (),
        |_, _| {
            let value = next_value.get();
            next_value.set(value + 1);
            g.set_event_value(&o.event, value)
        },
        drop,
    );

    // The bytes a frame reads back and writes in, with no GPU work writing them: Metal's
    // `getBytes` and `contents` wait for nothing, and nor may ours.
    group(
        g,
        "read_buffer_contents_256",
        (256, w),
        || vec![0u8; 256],
        |bytes: &mut Vec<u8>, _| g.read_buffer(&o.input, bytes),
        drop,
    );
    group(
        g,
        "get_bytes_256",
        (4, w),
        || vec![0u8; SMALL_BYTES],
        |bytes: &mut Vec<u8>, _| {
            g.get_bytes(
                &o.small_texture,
                (SMALL_SIDE, SMALL_SIDE),
                bytes,
                SMALL_SIDE * 4,
            )
        },
        drop,
    );
    group(
        g,
        "replace_region_256",
        (4, w),
        || (),
        |_, _| {
            g.replace_region(
                &o.small_texture,
                (SMALL_SIDE, SMALL_SIDE),
                &o.texels,
                SMALL_SIDE * 4,
            )
        },
        drop,
    );

    // Nothing above may have made a pipeline: the groups bind states made before them, and a
    // making is pipeline-creation time, which has its own ceiling and no place in a call's cost.
    let made_after = g.work_counts();
    for (counted, before) in &made_before {
        if *counted != "variant_pipelines_made" {
            continue;
        }
        let now = made_after
            .iter()
            .find(|(name, _)| name == counted)
            .map_or(0.0, |(_, count)| *count);
        emit(
            "percall",
            "meta_pipelines_made",
            "count",
            &mut [now - before],
        );
    }
}
