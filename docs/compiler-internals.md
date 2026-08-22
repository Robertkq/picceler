# Picceler Compiler Internals: The Pass Pipeline

This document explains how the compiler gets a `.pic` source file all the way down to LLVM IR:
how the initial `picceler`-dialect MLIR is built, what passes run on it, in what order, and why
that order matters.

**Audience:** contributors working on the compiler's front end or lowering passes. The pass
sections assume familiarity with MLIR concepts (dialects, dialect conversion, `RewritePattern`s,
`ConversionTarget`).

For what the ops/types themselves mean, see [`docs/dialect-reference.md`](dialect-reference.md) —
this document is about pipeline *structure*, not op-level semantics.

## 1. Front End: From `.pic` Source to Initial MLIR

Before any pass gets involved, three stages turn source text into the `picceler`-dialect MLIR
module the pass manager will run on:

1. **Lexer** (`src/lexer.cpp`) — turns the raw source text into a flat stream of `Token`s
   (identifiers, keywords, numbers, strings, symbols), skipping whitespace and `#` comments as it
   scans. `Lexer::getTokens()` drives this until it produces an `EOF_TOKEN`.
2. **Parser** (`src/parser.cpp`) — a recursive-descent parser that consumes the token stream and
   builds an AST (node types in `include/ast.h` / `src/ast.cpp`: `ModuleNode`, `FunctionNode`,
   `AssignmentNode`, `IfNode`, `ForNode`, `CallNode`, `BinaryOpNode`, ...). Expression parsing
   follows the precedence chain documented in [`LANGUAGE.md`](../LANGUAGE.md#operators--expression-precedence)
   (`parseRelational` → `parseAdditive` → `parseMultiplicative` → `parsePrimary`). Once parsing
   succeeds, `ModuleNode::normalizeTopLevelStatements()` applies the implicit-`main`-wrapping rule
   (also documented in `LANGUAGE.md`).
3. **MLIRGen** (`src/mlir_gen.cpp`) — walks the normalized AST and emits the initial
   `picceler`-dialect MLIR: one `func.func` per `FunctionNode`, and a `picceler.*` op for every
   builtin call (dispatched through `_functionTable`, the same table that backs the builtin lists
   in `LANGUAGE.md`).

The module MLIRGen produces is what `IRPassManager::run()` receives — Phase 1 below is the first
pass to touch it.

## 2. Pipeline Overview

```
.pic source
     │
     ▼   Lexer (src/lexer.cpp)
tokens
     │
     ▼   Parser + AST normalization (src/parser.cpp, src/ast.cpp)
AST
     │
     ▼   MLIRGen (src/mlir_gen.cpp)
picceler dialect MLIR
     │
     ▼
┌───────────────────────────┐
│ 1. High-Level Optimization │   addHighLevelOptimizationPasses()
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 2. Runtime Lowering        │   addRuntimeLoweringPasses()
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 3. Affine Lowering         │   addAffineLoweringPasses()
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 4. Backend Lowering        │   addBackendLoweringPasses()
└───────────────────────────┘
        │
        ▼
LLVM dialect → LLVM IR → object code
```

The four phases are registered in `IRPassManager::addPasses()` (`src/pass_manager.cpp`) and always
run in this order.

## 3. Phase 1 — High-Level Optimization

| # | Pass | Summary |
| --- | --- | --- |
| 1 | `mlir::createCanonicalizerPass()` | Cleans up the IR fresh out of `mlir_gen.cpp` before the pattern-matching pass below runs on it. |
| 2 | `PiccelerFiltersToConvPass` (`src/picceler_filters_to_conv_pass.cpp`) | Rewrites `sharpen` / `box_blur` / `gaussian_blur` / `edge_detect` / `emboss` into a canonical `picceler.convolution` + `picceler.kernel.const` pair, with kernel weights computed host-side (`calculateSharpenKernel`, `calculateBoxBlurKernel`, `calculateGaussianKernel`, ... in the same file). Collapses five op-specific lowerings into one, so every later pass only has to know how to lower `convolution`. |

After this phase, the only "filter" op left in the IR is `picceler.convolution`.

## 4. Phase 2 — Runtime Lowering

| # | Pass | Summary |
| --- | --- | --- |
| 1 | `PiccelerOpsToFuncCallsPass` (`src/picceler_ops_to_func_calls_pass.cpp`) | Converts the I/O-facing ops (`load_image`, `save_image`, `show_image`, `read_number`, `read_string`, `print`) into calls into the C runtime, and simultaneously starts converting `picceler.image` into `memref<?x?x4xi8>` via a `TypeConverter`, propagating the new type through function signatures, calls, and returns. Runs before affine lowering because Phase 3's patterns pattern-match on memref-typed operands, not the original `picceler.image` type. |

One nuance worth knowing: `print`'s format string must resolve to a compile-time
`picceler.string.const` — the pass splits it on `{}` at pass-run time and emits one runtime call
per literal chunk / substituted argument, so a `print` whose format string is itself a runtime
value will fail this pass.

## 5. Phase 3 — Affine Lowering

| # | Pass | Summary |
| --- | --- | --- |
| 1 | `PiccelerKernelToMemrefPass` (`src/picceler_kernel_to_memref_pass.cpp`) | Materializes `picceler.kernel.const`'s dense-attribute payload into a `memref.alloca` buffer via `memref.store`. Must run before `PiccelerToAffinePass`, since the convolution lowering pattern expects to `memref.load` kernel weights from a buffer, not read a constant attribute directly. |
| 2 | `PiccelerToAffinePass` (`src/picceler_to_affine_pass.cpp`) | The core compute-generation pass. Everything before it is type/op bookkeeping; everything after it is generic dialect-to-LLVM lowering. |

`PiccelerToAffinePass` rewrites each remaining compute op into `affine.parallel` loops over image
rows/columns operating directly on memref pixel buffers, using a separate conversion pattern per
op *shape* (grouped by the same TableGen interfaces used in `docs/dialect-reference.md`):

* `ElementWiseUnaryOpToAffine` — ops with `ElementWiseUnaryOpInterface`: `brightness`, `invert`.
* `RotateToAffine` — `rotate`, its own pattern (dimensions may swap for 90°/270° rotations, which
  doesn't fit the generic elementwise/neighborhood shapes).
* `NeighbourhoodOpsToAffine` — ops with `NeighbourhoodOpInterface`: `convolution` (how
  `sharpen`/`box_blur`/`gaussian_blur`/`edge_detect`/`emboss` reach this pass, per Phase 1),
  `dilate`, `erode`. Emits an `scf.if` per sampled neighbor to guard out-of-bounds reads at the
  image border.
* `ElementWiseBinaryOpToAffine` — ops with `ElementWiseBinaryOpInterface`: `diff`, `blend`. Emits
  an `scf.if` that checks the two input images have matching dimensions.
* `CropToAffine` — `crop`, its own pattern (copies a sub-rectangle rather than transforming every
  pixel of the input).

By the time this pass finishes, no Picceler compute op should remain — only `affine`, `arith`,
`memref`, and the occasional `scf.if` guard.

## 6. Phase 4 — Backend Lowering

This phase has no more Picceler-specific compute lowering to do — it's entirely about reaching the
LLVM dialect:

| # | Pass | Summary |
| --- | --- | --- |
| 1 | `PiccelerToLLVMIRPass` (`src/picceler_to_llvm_ir_pass.cpp`) | The last Picceler-specific pass: converts `picceler.string.const` into `llvm.global` + `llvm.addressof` + `llvm.gep`, and lowers `func` ops to LLVM. Marks the whole Picceler dialect *and* the `func` dialect illegal — the checkpoint that verifies Phases 1–3 left nothing behind. |
| 2 | `createReconcileUnrealizedCastsPass` (1st) | Folds away `UnrealizedConversionCastOp` bridges left by the type-converter passes above (e.g. the memref-descriptor cast built by `buildImageMemref` in Phase 2), before the structural lowerings below have to reason about them. |
| 3 | `createCanonicalizerPass` | General cleanup ahead of the structural lowerings. |
| 4 | `createLowerAffinePass` | Lowers `affine.for` / `affine.parallel` / `affine.apply` to `scf` + `arith` — there's no direct affine→LLVM path in upstream MLIR. |
| 5 | `createFinalizeMemRefToLLVMConversionPass` | Converts memref descriptors/ops into raw LLVM struct manipulation and `llvm.gep`/`load`/`store`. |
| 6 | `createSCFToControlFlowPass` | Lowers the `scf.if` from Phase 3's bounds/dimension-mismatch guards, plus whatever step 4 introduced, into unstructured `cf` branches. Must come after step 4, which is what produces most of the `scf` ops. |
| 7 | `createArithToLLVMConversionPass`, `createConvertControlFlowToLLVMPass`, `createConvertFuncToLLVMPass` | Lower `arith`, `cf`, and any remaining `func` ops to LLVM. `func` ops are normally already gone by step 1, so the `func`-to-LLVM pass here is a no-op safety net. |
| 8 | `createReconcileUnrealizedCastsPass` (2nd) | Final cleanup — steps 4–7 each introduce their own casts converting operand/result types piecemeal. After this the IR should be fully LLVM dialect, ready for translation to LLVM IR. |

**Note:** unlike MLIR pipelines that use `memref.alloc`, this one never heap-allocates a memref —
kernel buffers are `memref.alloca` (Phase 3, stack-allocated) and image buffers are built directly
around a pointer the C runtime already owns (Phase 2's `buildImageMemref`). That's why there's no
buffer-deallocation pass anywhere in this pipeline.

## 7. Debugging the Pipeline

`IRPassManager`'s constructor (`src/pass_manager.cpp`) wires up three pieces of built-in tooling
before any pass runs:

* **Full-IR dump to `dump.mlir`** — writes the module IR to `dump.mlir` (relative to the working
  directory the compiler was invoked from) after *every* pass, unbuffered, so a crash mid-compile
  still leaves a usable partial dump.
* **Per-pass IR-printing file tree** — additionally writes one file per pass invocation into
  `.pass_manager_output/` (MLIR's default tree directory for this API), handy when you want to
  jump straight to one specific pass's output instead of scrolling through `dump.mlir`.
* **`PassLogger` instrumentation** (`include/pass_manager.h`) — logs `Started pass: <name>` /
  `Finished pass: <name>` at `debug` level and `Failed pass: <name>` at `error` level via spdlog.

## 8. Adding a New Pass

Register the new pass in `IRPassManager::addPasses()` (`src/pass_manager.cpp`), inside whichever
`add*Passes()` helper matches what it operates on:

* Rewrites one high-level Picceler op into another, using only host-computable/constant data →
  `addHighLevelOptimizationPasses()` (see `PiccelerFiltersToConvPass` for the pattern to follow).
* Replaces a Picceler op with a runtime call, or extends the `picceler.image` → `memref` type
  conversion → `addRuntimeLoweringPasses()`.
* Generates the actual pixel-processing loops for a compute op → `addAffineLoweringPasses()`,
  after `PiccelerKernelToMemrefPass` if the new op consumes kernel data, and pick which of the
  four `*ToAffine` pattern shapes (unary / neighborhood / binary / bespoke like `crop`/`rotate`)
  the new op fits.
* Anything below the `affine`/`memref`/`scf` level, or standard MLIR-to-LLVM conversions →
  `addBackendLoweringPasses()`, keeping `PiccelerToLLVMIRPass` first since it's what enforces that
  no Picceler or `func` ops remain.

## See Also

* [`docs/dialect-reference.md`](dialect-reference.md) — op/type-level detail for every dialect
  that shows up during this pipeline.
