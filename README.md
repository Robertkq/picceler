# picceler

[![Unit Tests](https://github.com/Robertkq/picceler/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/unit_tests.yaml)
[![LIT MLIR Tests](https://github.com/Robertkq/picceler/actions/workflows/lit_mlir_tests.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/lit_mlir_tests.yaml)
[![E2E Tests](https://github.com/Robertkq/picceler/actions/workflows/e2e_tests.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/e2e_tests.yaml)
[![Publish Docs](https://github.com/Robertkq/picceler/actions/workflows/docs.yaml/badge.svg)](https://github.com/Robertkq/picceler/actions/workflows/docs.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

*picceler* (Pixel Accelerator) is a compiled domain specific language for image processing.
The language aims to simplify and fasten the development speed for image processing work (currently CPU-only, targeting the host's native architecture).

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

Picceler uses MLIR to go from a parsed `.pic` source file, through a custom `picceler` dialect, down through several lowering passes, to LLVM IR and finally native machine code.

Refer to the [Compiler Internals](docs/compiler-internals.md) document for a full breakdown of the pass pipeline (phases, order, and rationale), and to the [Dialect Reference](docs/dialect-reference.md) for op/type-level detail on the MLIR dialects involved.

# Profiling

Compile with `--profile` to automatically instrument every image operation and get a
Perfetto-viewable trace of where a program actually spends its time — no language changes
required. See [Profiling](docs/profiling.md) for usage, the `.bin` format, and how to convert a
trace for [ui.perfetto.dev](https://ui.perfetto.dev).

# Documentation

Generated API reference (Doxygen, rebuilt on every push to `main`): **[robertkq.github.io/picceler](https://robertkq.github.io/picceler/)**

