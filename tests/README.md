# Tests

Picceler has three independent test suites, each covering a different layer of the compiler:

| Suite | Directory | What it covers | Framework |
| --- | --- | --- | --- |
| Unit tests | [`unit/`](unit/) | Lexer/parser/AST, in isolation | GoogleTest |
| MLIR lit tests | [`lit/mlir/`](lit/mlir/) | Individual lowering passes, IR-to-IR | LLVM `lit` + `FileCheck` |
| E2E tests | [`lit/e2e/`](lit/e2e/) | Full pipeline: compile a `.pic` file, run the resulting executable, check its stdout | LLVM `lit` + `FileCheck` |

All three are enabled by default (`ENABLE_TESTS=ON`, `ENABLE_E2E_TESTS=ON` — see [BUILD.md](../BUILD.md) for the full list of build options) and run in CI on every pull request (`.github/workflows/unit_tests.yaml`, `lit_mlir_tests.yaml`, `e2e_tests.yaml`).

`mlir/` and `e2e/` share one lit config root (`tests/lit/lit.cfg.py`), with `e2e/lit.local.cfg` layering `.pic`-specific settings on top for its subtree. Both get copied into the build tree by `tests/lit/CMakeLists.txt`, landing at `build/mlir/` and `build/e2e/` respectively — so from a build directory it's `lit -v ./mlir` and `lit -v ./e2e`, not a path under `build/tests/`. `build/tests/lit/`, `build/tests/e2e/`, etc. only exist because CMake mirrors the source tree for every `add_subdirectory()` — they hold CMake's own bookkeeping, not the actual test content.

## Unit tests (`tests/unit/`)

GoogleTest-based tests of the lexer, parser, and AST, with no MLIR/LLVM involved. Build and run the `unittests` binary from your build directory:

```bash
cmake --build . --target unittests -j
./unittests
```

Add a new test by adding a `TEST_F`/`TEST` case to the relevant file in `tests/unit/src/` (`lexer_tests.cpp`, `parser_tests.cpp`, ...).

## MLIR lit tests (`tests/lit/mlir/`)

Each case is an `.mlir` file with `RUN:`/`CHECK:` lines that invoke `picceler-opt` with a specific pass and check the resulting IR. These test one pass in isolation — they don't compile or run an actual program.

```bash
cmake --build . --target mlir
```

This builds `picceler-opt`, stages the test files, and runs `lit` in one step. Equivalent to `cmake --build . -j && lit -v ./mlir` if you want to run `lit` yourself (e.g. to pass extra flags).

Add a new case by dropping an `.mlir` file into `tests/lit/mlir/`, following the existing files' `// RUN: %picceler-opt --<pass-name> ... | FileCheck %s` convention.

## E2E tests (`tests/lit/e2e/`)

Each case is a full `.pic` program with `RUN:`/`CHECK:` lines that (a) compile it with the built `picceler` binary, (b) run the resulting executable, and (c) check its stdout. This is the only suite that exercises the entire pipeline end-to-end, including the runtime library and actual generated machine code — the other two suites can't catch a case where the IR looks right at every intermediate stage but the compiled program itself crashes or misbehaves.

This suite checks stdout text, not pixel output. See [lit/e2e/README.md](lit/e2e/README.md) for exactly what that does and does not cover.

```bash
cmake --build . --target e2e
```

This builds `picceler`, stages the test files, and runs `lit` in one step. Equivalent to `cmake --build . -j && lit -v ./e2e` if you want to run `lit` yourself.

Every case has its `show_image()` calls stripped (it's a real blocking OpenCV window — it would hang in a headless CI container) and ends with a `print("<case>: ok\n")` plus a matching `CHECK:` line, so a successful run has an unambiguous signal even for cases that don't otherwise produce meaningful stdout. The two interactive cases (`print.pic`, `read_input.pic`) feed deterministic input via `printf "...\n" | %t`.

Add a new case by dropping a `.pic` file into `tests/lit/e2e/`, following the existing files' two-line `RUN:` convention:

```
# RUN: %picceler -o %t %s
# RUN: %t | FileCheck %s

# CHECK: <case-name>: ok

... your program, ending with print("<case-name>: ok\n") ...
```

Two things specific to this suite, useful if something breaks:

* **`img/cat.png`, `img/blue.png`, `img/white.png`** are copied into `build/e2e/img/` alongside the test cases (see `tests/lit/CMakeLists.txt`), so cases reference them as the relative path `img/<name>.png` — not the repo-root `img/` directory.
* **`main` must return `int64`** (or declare no return type at all, in which case a `return 0` is synthesized) — the compiler rejects any other declared return type for `main`, since its return value becomes the process exit code.

Also worth knowing about `.pic` files specifically: a file whose very first byte is `#` fails to parse (a real lexer bug — see the issue tracker), so every case here opens with a blank line before its `RUN:` header comment.
