# Profiling: `--profile`, the trace format, and Perfetto

Compile with `--profile` and a picceler binary records how long each `picceler` op took, and emits
a trace you can load into [Perfetto](https://ui.perfetto.dev). 

## Using it

```bash
./picceler --profile -o myProgram myProgram.pic
./myProgram
```

Running the binary writes `picceler_profiling_trace.bin` (cwd) on normal exit. Convert it and load
it in Perfetto:

```bash
python3 tools/picceler-trace-to-json/pictrace.py picceler_profiling_trace.bin -o trace.json
```

Open [ui.perfetto.dev](https://ui.perfetto.dev) and load `trace.json`. Each instrumented op is a
slice named after its MLIR op (e.g. `picceler.gaussian_blur`).

Without `--profile`, nothing changes: no buffer, no atexit handler, no file.

## What gets instrumented

Every `picceler` op, after initial canonicalization and before any other optimizations, except:

* `show_image`, `read_number`, `read_string` -- these block on a window/stdin, so one call would
  dwarf every real op's time on the timeline.
* `print` -- times stdout buffering, not compute.
* `kernel.const` -- sub-microsecond next to any real op; just adds a row.
* `string.const` -- instrumentation itself creates these to hold each traced op's name.

`load_image`/`save_image` are still instrumented: image decode/encode is genuinely part of the
timeline, often the dominant cost in a short pipeline.

## `--profile` overhead at `-O2`

`piccelerTraceBegin`/`piccelerTraceEnd` are opaque external calls, so LLVM must assume they clobber
arbitrary memory -- they act as optimization barriers around every instrumented op, blocking
cross-op LICM, loop fusion, and inlining across that boundary. They sit outside each op's own loop
nest, though, so inner-loop vectorization is unaffected. Measured with `bench/pic/bench_gaussian_blur.pic`
(50 iterations, `-O2`, plain wall clock, median of 3 runs): 64.38s without `--profile` vs. 62.80s
with it -- no measurable overhead on a compute-dominated program, since instrumentation is one pair
of calls per op invocation, not per pixel. Expect this to matter more on short/cheap ops or tight
loops of many small ops, where the per-call trace cost is a bigger fraction of the work being
measured.

## The `.bin` format

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

`tools/picceler-trace-to-json/pictrace.py` (stdlib only) validates magic/version,
turns each B/E pair into two JSON events, normalizes timestamps to start at `ts: 0`, and converts
nanoseconds to the microsecond float `ts` the format expects. `trackId` maps to `"tid"` as
`trackId + 1`; `"pid"` is always `1`. An unmatched begin/end (see the `abort()` caveat below) prints
a warning to stderr rather than producing broken JSON.

## Known limitation: `abort()` loses the trace

`atexit` doesn't run on `abort()`/`_exit()`/a fatal signal. `PiccelerToAffinePass` emits
`func.call @abort` for runtime-detected errors

