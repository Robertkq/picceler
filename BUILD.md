This document aims to provide all the necessary information to get picceler to build on your system.

We recommend building on Linux, and the following commands assume you are running Linux, but it is likely for picceler to build on Windows as well.


# Dependencies

We currently have the following dependencies: 
* **MLIR** - version 21.0
* **opencv** -- most versions should work
* **spdlog** - most versions should work
* **googletest** - most versions should work
* **CLI11** - most versions should work

For **spdlog**, **googletest** and **opencv** you can likely install them via your system's package manager.

```bash
sudo dnf install spdlog gtest opencv
```

Otherwise, you can manually build & install these libraries

**googletest**
```bash
git clone https://github.com/google/googletest.git && cd googletest
mkdir build && cd build
cmake .. && cmake --build . -j
sudo cmake --install .
```

**spdlog**
```bash
git clone https://github.com/gabime/spdlog.git && cd spdlog
mkdir build && cd build
cmake .. && cmake --build . -j
sudo cmake --install .
```
**opencv**
```bash
git clone https://github.com/opencv/opencv.git && cd opencv
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build . -j
sudo cmake --install .
```

For **MLIR** and **CLI11** you are much less likely to have them available via your system's package manager so we need to build them manually:

**CLI11**
```bash
git clone https://github.com/CLIUtils/CLI11.git && cd CLI11
mkdir build && cd build
cmake .. && cmake --build . -j
sudo cmake --install .
```

**MLIR**  

Building MLIR is expensive and will take you a long time compared to the other dependencies. Here are some tips to make it faster:

1. Make sure you have `Ninja` installed on your system. It is considerably faster for building MLIR
2. Make sure you have `clang` and `lld`/`mold` installed on your system. It is considerably faster than GNU ld
> The command below expects you have `Ninja` & `lld` installed. Modify accordingly if not installed
3. Leaving `-j` empty will give you the fastest compilation configuration, but it can make your PC struggle to do anything else besides compiling MLIR, you can set a value to `-j` to limit parallel jobs
```bash
git clone https://github.com/llvm/llvm-project.git --branch release/21.x --depth=1 && cd llvm-project # only get the branch we use!
mkdir build && cd build
cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INCLUDE_UTILS=ON \
        -DLLVM_USE_LINKER=lld \
        -DCMAKE_INSTALL_PREFIX=/usr/local 
cmake --build . -j
sudo cmake --install .
```
> Make sure you do not skip on any of these cmake options, otherwise your MLIR installation is ill-formed.

### Hooray, you are done! Now let's compile picceler!

```bash
git clone https://github.com/Robertkq/picceler.git && cd picceler
mkdir build && cd build
cmake .. && cmake --build . -j
```

If compilation was successful, `./picceler` is your compiler driver for `.pic` source files
(usage below; see [LANGUAGE.md](LANGUAGE.md) for the language itself). All binaries build
straight into `build/`, alongside it: `./picceler-opt` runs standalone passes over `.mlir` files
for testing, and `./picceler-mlir-lsp-server` backs editor tooling for the `picceler` dialect.

```bash
./picceler -o myExecutable ./examples/<file>.pic
./myExecutable # Try running it!
```

## Build options

These are set with `-D<OPTION>=<ON|OFF>` at the `cmake ..` configure step, e.g.
`cmake .. -DENABLE_DOCS=ON`.

| Option | Default | What it does |
| --- | --- | --- |
| `ENABLE_TESTS` | `ON` | Adds the `tests/` subdirectory (unit tests + MLIR lit tests, see "Running tests" below). |
| `ENABLE_E2E_TESTS` | `ON` | Adds the e2e lit suite (`tests/lit/e2e/`, the `e2e` build target). Set to `OFF` to skip building/copying it, e.g. if `lit` isn't available. |
| `ENABLE_CLANG_TIDY` | `OFF` | Runs `clang-tidy` as part of the normal build (`-warnings-as-errors=*`), using the project's `.clang-tidy` config. Requires `clang-tidy` to be on your `PATH`. |
| `ENABLE_DOCS` | `OFF` | Adds a `doc_doxygen` build target that generates the Doxygen API docs into `docs/html/` (requires Doxygen to be installed). Build it explicitly with `cmake --build . --target doc_doxygen`. |

### Build type

If you don't pass `-DCMAKE_BUILD_TYPE`, the project defaults to `RelWithDebInfo`. The usual CMake
build types are supported: `Debug`, `Release`, `RelWithDebInfo`.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

## Running tests

With the defaults (`ENABLE_TESTS=ON`, `ENABLE_E2E_TESTS=ON`), three independent test suites are
built — see [tests/README.md](tests/README.md) for what each one covers and how to add a case.

**Unit tests** (GoogleTest, `tests/unit/`) — build and run the `unittests` binary:

```bash
cmake --build . --target unittests -j
./unittests
```

**MLIR lit tests** (`tests/lit/mlir/`) and **e2e tests** (`tests/lit/e2e/`) — both use `lit`/`FileCheck`
(the LLVM test runner needs to be on your `PATH`), and both have a CMake target that builds
whatever they need and runs `lit` in one step:

```bash
cmake --build . --target mlir
cmake --build . --target e2e
```

All three suites run in CI on every pull request (`.github/workflows/unit_tests.yaml`,
`lit_mlir_tests.yaml`, `e2e_tests.yaml`).

## Install picceler on your system - WIP

There is currently no `install()` rule wired up in the CMake configuration, so `cmake --install .`
is a no-op today — this section is a placeholder for that work. For now, run the built binaries
directly out of `build/`, or add `build/` to your `PATH` yourself.
