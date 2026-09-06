#!/usr/bin/env python3
"""Converts a picceler profiling trace (.bin, written by picceler::TraceSession) into Chrome
Trace Event JSON, loadable directly in Perfetto (ui.perfetto.dev) or chrome://tracing.

See docs/profiling.md for the .bin format spec and how to produce a trace with --profile.
"""

import argparse
import json
import struct
import sys

# Must match traceMagic / traceVersion in lib/src/trace.cpp; see docs/profiling.md for the
# authoritative format spec.
MAGIC = 0x50494354  # "PICT"
SUPPORTED_VERSION = 1
EVENT_SIZE = 24

HEADER_FORMAT = "<IIQQ"  # magic, version, eventCount, eventSize
EVENT_FORMAT = "<QQIHBB"  # timestampNs, nameOffset, opIndex, trackId, phase, pad


class TraceFormatError(Exception):
    pass


def read_exact(data, offset, size, what):
    if offset + size > len(data):
        raise TraceFormatError(f"truncated trace file: could not read {what}")
    return data[offset : offset + size]


def parse_trace(data):
    header_size = struct.calcsize(HEADER_FORMAT)
    magic, version, event_count, event_size = struct.unpack(
        HEADER_FORMAT, read_exact(data, 0, header_size, "header")
    )
    offset = header_size

    if magic != MAGIC:
        raise TraceFormatError(f"bad trace file: magic 0x{magic:08x} does not match expected 0x{MAGIC:08x}")
    if version != SUPPORTED_VERSION:
        raise TraceFormatError(
            f"unsupported trace version: file is version {version}, this tool only supports version {SUPPORTED_VERSION}"
        )
    if event_size != EVENT_SIZE:
        raise TraceFormatError(
            f"corrupt trace file: event record size {event_size} does not match expected {EVENT_SIZE} for version {SUPPORTED_VERSION}"
        )

    (table_size,) = struct.unpack("<Q", read_exact(data, offset, 8, "string table size"))
    offset += 8
    string_table = read_exact(data, offset, table_size, "string table")
    offset += table_size

    events = []
    for i in range(event_count):
        raw = read_exact(data, offset, EVENT_SIZE, f"event {i}")
        offset += EVENT_SIZE
        timestamp_ns, name_offset, op_index, track_id, phase, _pad = struct.unpack(EVENT_FORMAT, raw)
        if name_offset >= len(string_table):
            raise TraceFormatError(f"corrupt trace file: event {i} has out-of-range name offset {name_offset}")
        end = string_table.find(b"\x00", name_offset)
        if end == -1:
            raise TraceFormatError(f"corrupt trace file: event {i} name is not null-terminated")
        name = string_table[name_offset:end].decode("utf-8", errors="replace")
        events.append(
            {
                "timestamp_ns": timestamp_ns,
                "name": name,
                "op_index": op_index,
                "track_id": track_id,
                "phase": chr(phase),
            }
        )

    return events


def check_balance(events):
    """Warns about unbalanced B/E pairs rather than rejecting the trace. Most commonly caused by
    the traced program hitting abort() mid-run, which skips the atexit flush for anything still
    in flight (see docs/profiling.md)."""
    open_by_track = {}
    for event in events:
        track = event["track_id"]
        stack = open_by_track.setdefault(track, [])
        if event["phase"] == "B":
            stack.append(event)
        elif event["phase"] == "E":
            if not stack:
                print(
                    f"warning: unbalanced trace: end event for op index {event['op_index']} "
                    f"('{event['name']}') on track {track} has no matching begin",
                    file=sys.stderr,
                )
            else:
                stack.pop()

    for track, stack in open_by_track.items():
        for event in stack:
            print(
                f"warning: unbalanced trace: begin event for op index {event['op_index']} "
                f"('{event['name']}') on track {track} has no matching end "
                "(the traced program may have hit abort())",
                file=sys.stderr,
            )


def to_chrome_trace_json(events):
    check_balance(events)

    baseline_ns = min((event["timestamp_ns"] for event in events), default=0)

    trace_events = []
    for event in events:
        trace_events.append(
            {
                "name": event["name"],
                "ph": event["phase"],
                # Chrome Trace Event "ts" is microseconds as a float; the .bin stores nanoseconds.
                "ts": (event["timestamp_ns"] - baseline_ns) / 1000.0,
                "pid": 1,
                "tid": event["track_id"] + 1,
            }
        )

    return {"traceEvents": trace_events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the .bin trace file (default: picceler_profiling_trace.bin)")
    parser.add_argument("-o", "--output", help="Path to write JSON to (default: stdout)")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = f.read()

    try:
        events = parse_trace(data)
    except TraceFormatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    doc = to_chrome_trace_json(events)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(doc, f)
    else:
        json.dump(doc, sys.stdout)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
