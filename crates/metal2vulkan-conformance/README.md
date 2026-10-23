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
| `work`, `runtime_work` | The copy-and-fill vocabulary a case states as data, and the runtime side that runs it |
| `measured` | The committed Metal measurements a ported case reads, under the public fixtures |
| `cases` | The case set itself, one module per family |

## The case set

| Family | What it compares | Metal's answer |
|---|---|---|
| `pass_bytes` | The words a compute pass wrote, exhaustively over a small input space | run on Metal |
| `math_routines` | The words `frexp` and `ilogb` write, scalar and per lane, over both widths | run on Metal |
| `stage_in_regions` | How a stage-in region clamps a dispatch's indices | run on Metal |
| `extended_range_planes` | A two-plane extended-range texture joined into one and split back again | run on Metal |
| `texture_buffer_fills` | Which bytes a fill changed over a texture made on a buffer | carried; `fillTexture:` is Apple SPI |
| `texture_fills`, `sample_fills` | Which bytes a texture fill, and a multisample sample fill, changed | carried; `fillTexture:` is Apple SPI |
| `two_plane_heap_blits`, `two_plane_copies` | What a blit or a copy moved across a two-plane surface | carried; the surface calls are Apple SPI |
| `mesh_draws` | The texels each mesh threadgroup's primitive drew, and the grids Metal draws nothing for | run on Metal |
| `mesh_barriers` | Whether a reader draw sees a writer draw's words across a memory barrier | run on Metal |
| `mesh_commands` | What an indirect command buffer's own-state mesh commands drew | run on Metal |
| `surface_fills` | Which bytes one private texture fill changed over a surface's plane or a buffer | carried; `fillTexture:` is Apple SPI |
| `indirect_private_types` | An indirect render command's read-back for each primitive type | carried; the getters are private |

A carried answer is our own measurement of this chip, stated in the shape the case emits, and its
case forbids re-recording with the reason why. Record mode refuses such a case rather than
pretending it can ask Metal again.
