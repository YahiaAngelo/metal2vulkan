# metal2vulkan-conformance

The Metal conformance suite. It makes one sentence executable:

> We are a Metal implementation. Any difference from Metal is a bug, unless we announced it.

A case is written once, against one vocabulary, and run by either side: native Metal on macOS, and
`metal2vulkan-runtime` on whatever Vulkan device the host has. Each run emits a **transcript**: a
tree of named sites holding what a Metal program can observe. Metal's answer is **recorded** on a
Mac, once per case, and **replayed** anywhere — the replay leg needs no Metal at all.

Comparison is exact by default. A loosened field names either a degrade notice's code or an entry of
`docs/API.md`'s accepted differences, and the comparison fails where that entry is gone or the
notice did not fire.

## This crate is not the validation suite

The harvested corpus, the stored observations, the corpus index, the censuses and audits and the
executors all belong to `metal2vulkan-validation`. None of it is this crate's business, and this
crate depends on none of it. What it uses from outside is `metal2vulkan-runtime`, the translator's
public API, `metal2vulkan-devtools` for fixture paths and `llvm_assemble`, and, on macOS,
`objc2-metal`.

The suite's own cases are its **case set** (`src/cases/`). Its recordings are its own, in
`recordings/`, in its own format; they are not observations and they enter no census.

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
| `record`, `replay` | Taking Metal's answer, and comparing a run against it |
| `report` | Verdict counts and the matrix of case against venue |
