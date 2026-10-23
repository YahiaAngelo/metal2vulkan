# metal2vulkan-conformance

The Metal conformance suite. It makes one sentence executable:

> We are a Metal implementation. Any difference from Metal is a bug, unless we announced it.

A case is written once, against one vocabulary, and run by either side: native Metal on macOS, and
`metal2vulkan-runtime` on whatever Vulkan device the host has. Each run emits a **transcript**: a
tree of named sites holding what a Metal program can observe. Metal's answer is **recorded** on a
Mac, once per case, and **replayed** anywhere — the replay leg needs no Metal at all.

Comparison is exact by default. A loosened field names either a degrade notice's code or an entry of
`docs/API.md`'s accepted differences, and the comparison fails where that entry is gone or the
notice did not fire. A site may name several, because a notice is announced once per device per
kind: the difference is justified where any of them holds.

A site where plain Metal answered a call and we returned a typed error is a **refusal**, and no
allowance can make it anything else: it counts in its own verdict and is never compliant.

A replay holds the runtime to exact answers. `conformance replay --default-policy` replays under
the runtime's own defaults instead — the fast form of every speed-for-accuracy choice, which is
what a consumer runs — so the suite can say whether a difference is the policy's or the
implementation's.

## The known-refusals ledger

`known-refusals.tsv` holds the refusals we have written down as open gaps. Each row is an **open
bug with a todo behind it** — not an accepted difference, not permission for anything. It exists so
that one case holding a known gap visible does not make every run red, because a report that is
always red is a report nobody reads, and that would cost us the next real refusal.

It changes the **exit status** and nothing else. Every row is still printed as a `Refused` row, with
our error and with Metal's own words where its API validation refuses the call too, and still
counted in the `refused` number.

It is an expectation, not an allowlist, and `replay` holds it in **both** directions:

- a refusal the ledger does not list fails the run — a new gap appeared;
- a row the ledger lists that is no longer refused **also** fails the run, saying the gap looks
  fixed and the row must go.

That second direction is the point: closing a gap forces the ledger edit, so a row cannot outlive
the bug it stands for and the file cannot rot into a silencer. A row is held only against the cases
a run selected, so replaying one case does not read every other row as fixed.

## Two Metal oracles

Each recording holds **two** Metal facts, and never collapses them.

**Plain Metal is the behavioural oracle**, because that is how a consumer runs the runtime: its
transcript is what comparison uses, and matching it is what the suite asks for.

**The same case under Metal's own API validation layer is a second recorded fact**
(`Validated::{Ran, Stopped, NotRun}`). It is never compared with our behaviour.

**A validated stop is not Metal refusing the call.** Where plain Metal answers and we return a
typed error, that is a gap of ours whatever its validation layer thinks, so the row is `Refused`
and is not compliant. The validated fact may **document** such a refusal
(`Justification::MetalValidation`) — a refusal Metal's own validation also refuses reads
differently from one Metal is happy with, and the first is the better bug report — and it is held
to its evidence: the case names the words Metal's validation used, and the claim holds only where
the recording's message contains them. An unexercised leg documents nothing. So the claim buys an
explanation in the report, never a pass, and it cannot be made at all unless the validated leg
really ran.

`record` takes the validated leg in a **child process** of the command, under `MTL_DEBUG_LAYER=1`
with `MTL_DEBUG_LAYER_ERROR_MODE=nslog`. Two reasons, both measured rather than assumed:

- Metal reads `MTL_DEBUG_LAYER` as it initialises, so one process cannot run both legs.
- Metal's validation reports by assertion, and its **default** error mode ends the process.
  `nslog` is the one mode that reports every check and lets the call run, so the suite's own
  process never has the layer on and no assertion of Apple's can end a record run.

The child announces each call it makes on its standard error, so Metal's own output and those
markers interleave in order and each complaint is attributed to the call that provoked it. Nothing
is matched against Apple's wording, so a check in a release the suite has not seen is still
recorded. `NSLog`'s timestamp and process prefix are stripped, because a recording holds nothing
host-specific.

A Metal-less host reads the validated fact out of the recording like everything else, so the axis
is as portable as the rest of a case.

## The cost half: what our side did

The transcript says what a case **observed**. The cost half says what our implementation **did** to
produce it, over the same case set:

```sh
conformance cost --runs 8            # measure and print, write nothing
conformance cost --runs 8 --write    # put the bounds in the cases' recordings
conformance replay                   # hold each case to them, and report
```

A case's cost record lives in its own recording file, beside the Metal answers, so a host with no
Metal reads it with the rest of the case.

**Counts are ours, never Metal's.** Metal has no counterpart for any of them and a host with no
Metal can have no answer, so a count never enters the comparison with Metal and never produces a
correctness verdict. The cost half has its own three verdicts — `within`, `over`, `unmeasured` — its
own section of the report, and its own line of the command's exit status.

**It gates on counts, not times.** A count is deterministic, so it is assertable on any host under
any load, and a regression reads "this case now does two allocations where it did none" rather than
as a percentage nobody can reproduce. The mechanism is the runtime's own `WorkCounts`, read twice
around each case; nothing here counts anything the runtime does not already count, and a cost run
asks it for no more than an ordinary run does.

Work is attributed to three regions, kept apart because moving work between them is not a win:
**translation** (making a module, a layout or a pipeline — the one-time cost, which has its own
ceilings), **the device API** (what a call does on the calling thread), and **submission** (what the
commit path, our own other threads and the GPU do afterwards, so a call that returns at once while
queueing work cannot score as a win). The device API is the residual.

A count the measurement never saw move is bounded at **zero**, so a case that starts doing something
it did not do before is `over` rather than unnoticed.

Some counts are reported and never gated, because holding them would give a gate that fires at
random on honest work. Each reason is written at its site in `cost.rs` with the runs behind it:

- one whose readings **moved** between the measurement's own rounds, which the record keeps as the
  range it was seen in;
- one that says **which thread arrived first** — which thread replayed a deferred command buffer,
  when one of our background threads ran, and everything on the asynchronous release path, whose
  counts may land in a later case's window;
- one that is a reading of **what was alive at that instant** rather than work a case did;
- one that says an object was **found already made** rather than made, which depends on what every
  case before it in the same process made;
- `completed_waits`, which charges a case for waits belonging to whatever ran before it.

Totals stay gated where one exists, so nothing is lost: `command_buffers_deferred` holds the two
replay counts between them, and a `_made` count holds its `_shared` partner — finding one already
made only ever *lowers* it, and the gate is an upper bound.

**A bound is a bound on the selection it was measured over.** A case's counts depend on what ran
before it in the same process, so a cost record is evidence for a whole-set run and nothing else; a
run of fewer cases reads `unmeasured`. When cases land, re-measure. To tell a case's own work from
what the process handed it, run the contrast:

```sh
conformance cost --runs 2 --show <case>   # a selection of one, on its own fresh device
conformance cost --runs 1 --show          # the same case inside the whole set
```

A count that differs between the two depends on the selection. That is how `completed_waits` was
caught charging three cases for four or five waits that were not theirs, after it had held still
over seven rounds of an earlier measurement.

**A device's gap is not a cost win.** A count that is lower because the device lacks the exact form,
so the runtime announced a notice and did something cheaper, is the device's gap and not work we
saved. A record carries the notices its measurement announced and refuses to compare a run that
announced otherwise. Four things make a run `unmeasured` rather than a pass, each said in words: a
venue the measurement was not taken on, a selection that is not the whole case set (a case's counts
depend on what ran before it in the same process), a run that stopped part way, and a run the device
degraded otherwise.

Times are **advisory** here: a wall-clock figure is printed per case so a reader can see where the
counts are large, and it is neither recorded nor gated. Per-call timing against Metal is the per-call
cost table in `metal2vulkan-runtime/examples/host_ab`, which this half does not duplicate.

## This crate is not the validation suite

The harvested corpus, the stored observations, the corpus index, the censuses and audits and the
executors all belong to `metal2vulkan-validation`. None of it is this crate's business, and this
crate depends on none of it. What it uses from outside is `metal2vulkan-runtime`, the translator's
public API, `metal2vulkan-devtools` for fixture paths and `llvm_assemble`, and, on macOS,
`objc2-metal`.

The suite's own cases are its **case set** (`src/cases/`). Its recordings are its own, in
`recordings/`, in its own format; they are not observations and they enter no census.

A case ported from a runtime lane restates that lane's measured Metal values and reads the lane's
committed measurement files under `fixtures/public/`. Those are our own measurements over our own
fixtures, which is what the boundary allows; the lanes stay where they are and a ported case never
edits one.

## Layout

| Module | What it is |
|---|---|
| `transcript` | The record shape: a tree of sites, each a value, its shape and an optional digest |
| `allowance` | The justification a loosened field must name, checked against the degrade catalogue and `docs/API.md` |
| `compare`, `diff` | Field-by-field comparison, and the diff that names the leaf that moved |
| `verdict` | Exact, announced, divergent, refused, unrecorded, and per venue "lacks the device feature" |
| `recording` | A case's Metal answer, keyed by chip, macOS build and Metal level, with its runs and its outcome set |
| `case` | The case API and the vocabulary both sides answer in |
| `runtime_side` | `metal2vulkan-runtime` as a side |
| `metal_side` | Native Metal as a side (macOS only), which record mode runs |
| `validated` | The second oracle: the same case under Metal's API validation layer, in a child process |
| `record`, `replay` | Taking Metal's answer, and comparing a run against it |
| `report` | Verdict counts and the matrix of case against venue |
| `cost` | The cost half: the counts our side moved, their regions, their bounds and the three cost verdicts |
| `work`, `runtime_work`, `metal_work` | The copy-and-fill vocabulary a case states as data, and the two sides that run it |
| `measured` | The committed Metal measurements a ported case reads, under the public fixtures |
| `known_refusals` | The ledger of open gaps, held in both directions; it sets the exit status, never a verdict |
| `cases` | The case set itself, one module per family |

## The case set

| Family | What it compares | Metal's answer |
|---|---|---|
| `pass_bytes` | The words a compute pass wrote, exhaustively over a small input space | run on Metal |
| `math_routines` | The words `frexp` and `ilogb` write, scalar and per lane, over both widths | run on Metal |
| `stage_in_regions` | How a stage-in region clamps a dispatch's indices | run on Metal |
| `extended_range_planes` | A two-plane extended-range texture joined into one and split back again | run on Metal |
| `rgb10a2_srgb` | What a kernel reads, writes and samples of Apple's padded 10-bit sRGB format, and what a full-screen draw stores into it | run on Metal |
| `texture_buffer_fills` | Which bytes a fill changed over a texture made on a buffer | carried; `fillTexture:` is Apple SPI |
| `texture_fills`, `sample_fills` | Which bytes a texture fill, and a multisample sample fill, changed | carried; the per-format digests are the probes' own truncating formatter's, on an Apple M4 |
| `texture_fills`' heap runs | Every byte of the heap one two-plane fill changed, where the per-format digest elided the run | run on Metal, through the fill's SPI selector |
| `two_plane_heap_blits` | What a blit moved between two-plane textures placed in a heap | run on Metal |
| `two_plane_copies` | What a blit or a fill moved across a two-plane surface's rows | run on Metal where the case fills nothing; carried where it does (`fillTexture:` is Apple SPI) |
| `mesh_draws` | The texels each mesh threadgroup's primitive drew, and the grids Metal draws nothing for | run on Metal |
| `mesh_barriers` | Whether a reader draw sees a writer draw's words across a memory barrier | run on Metal |
| `mesh_commands` | What an indirect command buffer's own-state mesh commands drew, and what two commands with no barrier between them determine | run on Metal |
| `surface_fills` | Which bytes one private texture fill changed over a surface's plane or a buffer | carried; `fillTexture:` is Apple SPI |
| `indirect_private_types` | An indirect render command's read-back for each primitive type | carried; the getters are private |
| `pixel_samplers` | A pixel-coordinate sampler's samples, through a slot, an argument buffer, a normalized control and a constexpr sampler | run on Metal |
| `pixel_downscales` | A linear downscale by eight through a pixel-coordinate sampler, at seven texture sizes | run on Metal |
| `chroma_pairs` | What a copy of a 4:2:2 chroma-pair texture moved, and what a kernel's `read()` of one answers | run on Metal |
| `ycbcr_textures` | What Metal says about every two-plane YCbCr format's descriptors, and its `read()` of a `420v` surface's texels | the descriptors run on Metal; the texels carried, a YCbCr texture needs an IOSurface |
| `ycbcr_minify` | Where a minified 10-bit YCbCr texture's linear footprint puts the switch, along each axis | carried; a YCbCr texture needs an IOSurface |
| `compute_threadgroups` | What the threads of a 32-, 64- and 1024-thread threadgroup wrote through a simdgroup shuffle | run on Metal; the over-limit row carried from an Apple M4 |
| `buffer_blits` | Where a blit encoder's buffer fills and copies landed, overlapping ones included | run on Metal |
| `simdgroup_lockstep` | The threadgroup words two loops only a simdgroup's lockstep orders came to | run on Metal |
| `color_write_masks` | Which channels of a texel a draw changes under a write mask of none, some and all of them | run on Metal |

A carried answer is our own measurement of this chip, stated in the shape the case emits, and its
case forbids re-recording with the reason why. Record mode refuses such a case rather than
pretending it can ask Metal again.

**Apple SPI is not the same thing as absent.** `fillTexture:level:slice:region:bytes:length:` has
no public header, so `objc2-metal` binds nothing for it and the copy-and-fill vocabulary's pattern
fill was carried from `probes/`. `metal_work` reaches it by its selector after
`respondsToSelector:`, so a case built on it can be **run** rather than carried; the heap-run cases
above are the first. Two consequences worth stating plainly, because a stale reason in this table is
how `two_plane_copies` lost its Metal leg in the first place:

- `texture_fills`' per-format digests stay carried for a different reason than they used to. Not
  because the fill cannot be reached, but because each is a digest of twenty per-case answers of
  which only the outer digest was ever written down, and that one was taken on an Apple M4. Making
  those rows `MetalSource::Run` and recording per-case values on this chip is a packet of its own.
- `texture_buffer_fills`, and `two_plane_copies`' filling cases, say "`fillTexture:` is Apple SPI".
  That is still true of the words and no longer the whole reason: nothing has moved them yet. The
  colour fills (`fillTexture:...:color:` and `...:color:pixelFormat:`) have no arm in `metal_work`
  at all, so a case built on one is carried outright.

## `probes/`

Where a carried answer came from a call record mode cannot make — Apple SPI, a private getter —
the Objective-C probe that measured it is committed here, with the command that builds it in its
header comment. A probe deleted after one run is how a measurement evaporates: the
`RG10Unorm_X12_sRGB` row of `surface_fills` was missing for exactly that reason, because the
original probe was thrown away and the lane that kept its number printed only the first 24 bytes
of each 32-byte run. A probe that re-measures rows already carried, as a control, says whether its
new rows can be trusted.
