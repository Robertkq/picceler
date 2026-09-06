#!/usr/bin/env python3
"""POC driver: compiles+runs bench/pic/bench_*.pic with --profile for picceler's own numbers, runs
reference_bench for the naive C++ / OpenCV columns, and merges both into bench/RESULTS.md.

Usage: bench/run_bench.py [--build-dir build] [--image bench.jpg] [--iterations 50]
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "picceler-trace-to-json"))
import pictrace  # noqa: E402 (needs sys.path set up first)

OPERATIONS = ["invert", "brightness", "gaussian_blur", "sharpen"]
LABELS = {
    "invert": "invert",
    "brightness": "brightness(+30)",
    "gaussian_blur": "gaussian_blur(r=6)",
    "sharpen": "sharpen(3x3)",
}


def median_duration_ms(events, name):
    open_by_track = {}
    durations = []
    for event in events:
        stack = open_by_track.setdefault(event["track_id"], [])
        if event["phase"] == "B":
            stack.append(event)
        elif stack and stack[-1]["name"] == event["name"]:
            begin = stack.pop()
            if event["name"] == name:
                durations.append((event["timestamp_ns"] - begin["timestamp_ns"]) / 1e6)
    if not durations:
        raise RuntimeError(f"no trace events found for '{name}'")
    return statistics.median(durations)


def run_picceler_bench(picceler_bin, build_dir, pic_dir, work_dir, op, native, opt_level):
    pic_file = pic_dir / f"bench_{op}.pic"
    exe = work_dir / f"bench_{op}_bin"
    args = [str(picceler_bin), "--profile", "--opt-level", str(opt_level), "-o", str(exe), str(pic_file)]
    if native:
        args.append("--native")
    subprocess.run(args, check=True, cwd=build_dir)

    trace_path = build_dir / "picceler_profiling_trace.bin"
    trace_path.unlink(missing_ok=True)
    subprocess.run([str(exe)], check=True, cwd=build_dir, stdout=subprocess.DEVNULL)

    events = pictrace.parse_trace(trace_path.read_bytes())
    trace_path.unlink()
    return median_duration_ms(events, f"picceler.{op}")


def run_reference_bench(reference_bench_bin, image_path, iterations):
    result = subprocess.run(
        [str(reference_bench_bin), str(image_path), str(iterations), "--json"],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    return {r["operation"]: (r["naive_ms"], r["opencv_ms"]) for r in data["results"]}


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def render_results_md(picceler_results, reference_results, iterations, image_path):
    lines = [
        "# Benchmark Results",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- CPU: {cpu_model()}",
        f"- Image: `{image_path.name}` ({image_path.stat().st_size} bytes)",
        f"- Iterations: {iterations}",
        "",
        "Generated locally by `bench/run_bench.py`, not in CI -- see bench/reference_bench.cpp for "
        "the naive/OpenCV implementations and bench/pic/ for the picceler programs.",
        "",
        "| operation | picceler | naive C++ | OpenCV |",
        "| --- | --- | --- | --- |",
    ]
    for op in OPERATIONS:
        naive_ms, opencv_ms = reference_results[op]
        lines.append(f"| {LABELS[op]} | {picceler_results[op]:.1f} ms | {naive_ms:.1f} ms | {opencv_ms:.1f} ms |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--image", default="bench.jpg", help="filename under <build-dir>/img/")
    parser.add_argument("--iterations", type=int, default=50,
                        help="reference_bench iterations; the .pic programs' own loop count is fixed at compile "
                             "time (currently 50 too) -- change bench/pic/bench_*.pic to keep them in sync")
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "bench" / "RESULTS.md")
    parser.add_argument("--native", action="store_true",
                        help="pass --native to picceler, targeting the host CPU instead of the "
                             "generic baseline; the produced RESULTS.md is then only valid for this CPU")
    parser.add_argument("--opt-level", type=int, choices=[0, 1, 2, 3], default=2,
                        help="passed through as picceler's --opt-level (default 2, matching picceler's own default)")
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    picceler_bin = build_dir / "picceler"
    reference_bench_bin = build_dir / "reference_bench"
    image_path = build_dir / "img" / args.image
    pic_dir = REPO_ROOT / "bench" / "pic"
    work_dir = build_dir / "bench_tmp"
    work_dir.mkdir(exist_ok=True)

    for required in (picceler_bin, reference_bench_bin, image_path):
        if not required.exists():
            sys.exit(f"error: required path does not exist: {required}")

    print(f"Running naive C++ / OpenCV reference ({args.iterations} iterations)...")
    reference_results = run_reference_bench(reference_bench_bin, image_path, args.iterations)

    picceler_results = {}
    for op in OPERATIONS:
        print(f"Compiling and profiling picceler '{op}'...")
        picceler_results[op] = run_picceler_bench(picceler_bin, build_dir, pic_dir, work_dir, op, args.native,
                                                  args.opt_level)

    args.output.write_text(render_results_md(picceler_results, reference_results, args.iterations, image_path))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
