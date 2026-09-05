# Profiling: `--profile`, the trace format, and Perfetto

Compile with `--profile` and a picceler binary records how long each `picceler` op took, and emits
a trace you can load into [Perfetto](https://ui.perfetto.dev). No language changes — it's a
compiler flag, not a builtin.

## Using it

```bash
./picceler --profile -o myProgram myProgram.pic
./myProgram
```

Running the binary writes `picceler_profiling_trace.bin` (cwd) on normal exit. Convert it and load
it in Perfetto:

```bash
python3 tools/picceler-trace-to-json/picceler-trace-to-json.py picceler_profiling_trace.bin -o trace.json
```

Open [ui.perfetto.dev](https://ui.perfetto.dev) and load `trace.json`. Each instrumented op is a
slice named after its MLIR op (e.g. `picceler.gaussian_blur`).

Without `--profile`, nothing changes: no buffer, no atexit handler, no file.

## What gets instrumented

`PiccelerAddProfilingPass` runs right after the canonicalizer and before `PiccelerFiltersToConvPass`
(see [`docs/compiler-internals.md`](compiler-internals.md)) — early enough that ops still have the
name you wrote (`gaussian_blur`, not the `convolution` it later lowers into), and late enough that
dead/folded ops are already gone. It walks **every op in the `picceler` dialect except
`string.const`** — not just compute ops, also `load_image`, `print`, `kernel.const`, etc. —
and wraps each with a `piccelerTraceBegin` call before it and `piccelerTraceEnd` after, passing the
op's name, a `uint32` index (0, 1, 2... in walk order), and a `uint16` track id (always `0` today;
reserved for a future GPU/async queue). `string.const` is skipped: it's a compile-time constant
materialization (still near-zero cost even after lowering to an LLVM global), so instrumenting it
was mostly noise -- every string literal in a program got its own tiny slice.

The index is a compile-time constant per static op, not a per-execution id — an op inside a loop or
a function called from multiple places reports the same index on every dynamic firing. Perfetto
still renders each firing as its own slice (position comes from timestamp, not index), so this only
matters if you want to tell firings apart programmatically.

Instrumented ops keep MLIR's `Pure` trait, which in principle permits hoisting/CSE-ing them out from
between their own trace calls. In practice this pipeline never runs a CSE or loop-invariant-code-motion
pass anywhere, so it doesn't happen today — but nothing enforces that if a pass gets added between
Phase 1 and Phase 3, so re-check this if you touch that part of the pipeline.

## The `.bin` format

`lib/include/trace.h` / `lib/src/trace.cpp`: `TraceSession` is a singleton, created lazily on the
first event and deliberately never destroyed (it flushes from an `atexit` handler; leaking avoids a
possible use-after-free from destructor/atexit ordering being unspecified — see
`.lsan-suppressions`). Timestamps use `steady_clock`, not `system_clock`.

Layout (host-endian, no cross-arch portability implied):

| Field | Type | Notes |
| --- | --- | --- |
| magic | `uint32` | `0x50494354` ("PICT") |
| version | `uint32` | Currently `1`. |
| eventCount | `uint64` | |
| eventSize | `uint64` | `24` for version 1. |
| stringTableSize | `uint64` | |
| stringTable | `stringTableSize` bytes | Null-terminated names, deduplicated by content. |
| events | `eventCount * eventSize` bytes | See below. |

Each event (24 bytes):

| Field | Type | Notes |
| --- | --- | --- |
| timestampNs | `uint64` | |
| nameOffset | `uint64` | Offset into the string table. |
| opIndex | `uint32` | |
| trackId | `uint16` | |
| phase | `uint8` | `'B'` or `'E'` |
| pad | `uint8` | Always `0`. |

## Converting to Chrome Trace Event JSON

`tools/picceler-trace-to-json/picceler-trace-to-json.py` (stdlib only) validates magic/version,
turns each B/E pair into two JSON events, normalizes timestamps to start at `ts: 0`, and converts
nanoseconds to the microsecond float `ts` the format expects. `trackId` maps to `"tid"` as
`trackId + 1`; `"pid"` is always `1`. An unmatched begin/end (see the `abort()` caveat below) prints
a warning to stderr rather than producing broken JSON.

## Known limitation: `abort()` loses the trace

`atexit` doesn't run on `abort()`/`_exit()`/a fatal signal. `PiccelerToAffinePass` emits
`func.call @abort` for runtime-detected errors (e.g. `diff`/`blend` size mismatches,
`box_blur`/`gaussian_blur` out-of-range radius), so a `--profile` program that hits one loses
whatever was buffered — no partial trace. Accepted gap; no crash handler installed.
