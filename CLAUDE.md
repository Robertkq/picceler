Picceler is a compiled DSL for image processing, built to explore MLIR — see [README.md](README.md).

## Where things live

- **Language syntax & builtins**: [LANGUAGE.md](LANGUAGE.md)
- **Build, build options, running tests**: [BUILD.md](BUILD.md)
- **MLIR pass pipeline (front end → passes → LLVM IR)**: [docs/compiler-internals.md](docs/compiler-internals.md)
- **MLIR dialect/op/type reference**: [docs/dialect-reference.md](docs/dialect-reference.md)

Don't restate content from these docs here — update them instead, and keep this file to things an
agent needs on every task.

## Build & test (quick reference — see BUILD.md for the full picture)

```bash
mkdir build && cd build
cmake .. && cmake --build . -j   # binaries land directly in build/: picceler, picceler-opt, picceler-mlir-lsp-server, unittests
./unittests                      # GoogleTest unit tests (tests/unit/)
cmake --build . --target mlir    # MLIR lit tests (tests/lit/mlir/)
cmake --build . --target e2e     # e2e lit tests (tests/lit/e2e/) — compiles & runs real .pic programs
```

Default build type is `RelWithDebInfo`. `-DENABLE_CLANG_TIDY=ON` / `-DENABLE_DOCS=ON` are opt-in.

## Tech stack

C++23, CMake, LLVM/MLIR (21.0), OpenCV (image I/O in the runtime), spdlog (logging), CLI11
(compiler CLI), GoogleTest + LLVM `lit`/`FileCheck` (tests).

## Architecture, in one paragraph

`.pic` source → Lexer (`src/lexer.cpp`) → Parser (`src/parser.cpp`) builds an AST (`src/ast.cpp`)
→ MLIRGen (`src/mlir_gen.cpp`) emits the initial `picceler` MLIR dialect → four pass phases
(`src/pass_manager.cpp`) lower it to the LLVM dialect → LLVM IR → object code, linked against the
`PiccelerRuntime` static library (`lib/`, OpenCV-backed) for I/O. Full detail in
[docs/compiler-internals.md](docs/compiler-internals.md).

## Conventions

- **Naming**: namespaces `snake_case` (`picceler`), classes `PascalCase`, methods/functions
  `camelCase`, member variables `_camelCase`, local variables `camelCase`.
- **Memory**: smart pointers + RAII, avoid raw owning pointers.
- **Logging**: `spdlog`, not `std::cout`/`printf`.
- **Dialect changes**: define ops/types/interfaces in TableGen (`tablegen/*.td`), not hand-written C++.
- **New compute ops**: prefer lowering to MLIR (`affine`, or an existing `*ToAffine` pattern shape)
  over adding a new runtime-library call — the runtime is reserved for I/O and system-level tasks.
  See "Adding a New Pass" in [docs/compiler-internals.md](docs/compiler-internals.md).
- **Lit tests**: prefer structural `CHECK` assertions over exact full-IR snapshots for lowerings
  that are still likely to change.

## Things to double-check before trusting old context

- `cmake --install` currently has no `install()` rules wired up — it's a known no-op, not a bug to
  quietly "fix"; see BUILD.md.
- `docs/` is *not* fully gitignored — only `docs/html/` (generated Doxygen output) is. Don't assume
  files under `docs/` are untracked.
- `main()` must declare `-> int64` or no return type at all (implicit `return 0`) — any other
  declared return type is a compile error, since main's return value becomes the process exit code.
