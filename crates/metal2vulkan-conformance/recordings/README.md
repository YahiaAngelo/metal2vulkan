# Recordings

One file per case: Metal's answer, keyed by chip, macOS build and Metal level, with the runs behind
it and the set of outcomes where the behaviour varies.

A recording of another chip, macOS build, file format or suite version is stale. The suite refuses
it rather than comparing against it; nothing here ever passes quietly.

These are the conformance suite's own files in its own format. They are not the validation crate's
observations and they enter no census.

A **carried** answer is a Metal measurement this tree already holds, restated by the case in the
shape it emits, for a case Metal has no public call to answer: a texture fill, a two-plane
extended-range format, an indirect command's read-back. Its provenance names where it came from,
and where it was measured on another chip than the key's it says so there in words.
