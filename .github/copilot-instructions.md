# Project Context
**Picceler** is a custom DSL compiler for image processing, built to explore and learn MLIR.

## Architecture
- **Goal**: Compile DSL to optimized assembly using MLIR and LLVM.
- **Runtime**: I/O functions (`load`, `save`, `show`) translate to calls in the runtime library (`libpicceler_runtime.a` / `runtime.so`).
- **Operations**: Image processing operations (e.g., `brightness`, `blur`) are lowered to MLIR and compiled to native assembly for performance.
- **Pipeline**: Source -> Lexer/Parser -> AST -> MLIR Generation -> Passes/Optimization -> LLVM IR -> Object File.

## Tech Stack
- **Language**: C++ (Modern C++20 standards).
- **Build System**: CMake.
- **Core Libraries**: LLVM, MLIR, CLI11, spdlog, gtest.

# Coding Standards
- **Quality**: Prioritize high-quality, maintainable, and modern C++ code.
- **MLIR**: Follow MLIR best practices. Use TableGen (`.td` files) for defining dialects, operations, and types.
- **Memory Management**: Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) and RAII. Avoid raw pointers where possible.
- **Logging**: Use `spdlog` for all logging (info, debug, error).

# Naming Conventions
- **Namespaces**: `snake_case` (e.g., `picceler`).
- **Classes**: `PascalCase` (e.g., `Compiler`).
- **Methods/Functions**: `camelCase` (e.g., `emitObjectFile`).
- **Member Variables**: `_camelCase` (e.g., `_cliApp`, `_context`).
- **Variables**: `camelCase`.

# Specific Instructions for Copilot
- When suggesting MLIR code, ensure it aligns with the defined dialects in `include/ops.h` and `tablegen/`.
- If a task involves image operations, prefer
