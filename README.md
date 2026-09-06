# picceler

[![Unit Tests](https://github.com/Robertkq/picceler/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/unit_tests.yaml)
[![LIT MLIR Tests](https://github.com/Robertkq/picceler/actions/workflows/lit_mlir_tests.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/lit_mlir_tests.yaml)
[![E2E Tests](https://github.com/Robertkq/picceler/actions/workflows/e2e_tests.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/e2e_tests.yaml)
[![Publish Docs](https://github.com/Robertkq/picceler/actions/workflows/docs.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/docs.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

*picceler* (Pixel Accelerator) is a compiled domain specific language for image processing.
The language aims to simplify and fasten the development speed for image processing work (currently CPU-only).

Picceler doesn't aim to be a production language, rather it aims to provide a good example of what you can achieve with MLIR.

# How to build & install

The process of building and installing can be quite long, please refer to the [How to build](BUILD.md) guide for this information.

# Language

The picceler language is statically typed and immutable: every variable and function parameter
carries an explicit type annotation (`int64`, `float64`, `string`, `image`, `kernel`), and once a
name is declared it can't be reassigned. Keeping things simple.

Please refer to the [Language](LANGUAGE.md) document for more precise information on syntax and builtin operations.

An official VS Code extension is vendored as a submodule at
[`editors/vscode-picceler`](editors/vscode-picceler) — see its
[README](editors/vscode-picceler/README.md) for details.

# Inner workings

Picceler uses MLIR to go from a parsed `.pic` source file, through a custom `picceler` dialect, down through several lowering passes, to LLVM IR and finally native machine code. LLVM's own middle-end
optimization pipeline runs on the way to machine code; `-O0`-`-O3` (default `-O2`) controls it, and
`--native` targets the host CPU instead of a portable generic baseline.

Refer to the [Compiler Internals](docs/compiler-internals.md) document for a full breakdown of the pass pipeline (phases, order, and rationale), and to the [Dialect Reference](docs/dialect-reference.md) for op/type-level detail on the MLIR dialects involved.

# Performance

Picceler is measured against a hand written naive C++ loop and against
OpenCV, on the same image:

| operation | picceler | naive C++ | OpenCV |
| --- | --- | --- | --- |
| invert | 9.5 ms | 7.0 ms | 4.8 ms |
| brightness(+30) | 9.4 ms | 7.9 ms | 5.2 ms |
| gaussian_blur(r=6) | 1106.1 ms | 1039.5 ms | 12.1 ms |
| sharpen(3x3) | 72.4 ms | 78.8 ms | 15.3 ms |

Picceler started out 4 to 6 times slower than naive C++ on elementwise
operations, and 1.2 to 1.4 times slower on convolutions. Enabling
optimizations that were free to add brought that down: CSE, dead value
elimination, and loop invariant code motion in the MLIR passes, plus a real
LLVM optimization pipeline (`-O0` to `-O3`) and native CPU codegen
(`--native`). Elementwise operations now land within 1.2 to 1.4x of naive
C++, and sharpen is slightly faster than it.

Losing to OpenCV by a wide margin was always expected.

See [bench/RESULTS.md](bench/RESULTS.md) for full numbers and [bench/](bench/)
for how to reproduce them.

> I found out that building a compiler is hard, and comparing its output to
> production compilers and tech stacks is a harsh reality.

# Profiling

Compile with `--profile` to automatically instrument every image operation and get a
Perfetto-viewable trace of where a program actually spends its time — no language changes
required. See [Profiling](docs/profiling.md) for usage, the `.bin` format, and how to convert a
trace for [ui.perfetto.dev](https://ui.perfetto.dev).

# Documentation

Generated API reference (Doxygen, rebuilt on every push to `main`): **[robertkq.github.io/picceler](https://robertkq.github.io/picceler/)**

See [ROADMAP.md](ROADMAP.md) for what is intentionally out of scope for now.

