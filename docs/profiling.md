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

All `picceler` ops, after initial canonicalization, before any other optimisations otherwise

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

