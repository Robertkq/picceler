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
| 1.5 | `PiccelerAddProfilingPass` (`src/picceler_add_profiling_pass.cpp`) — **only when `--profile` is passed** | Wraps every `picceler`-dialect op with `piccelerTraceBegin`/`piccelerTraceEnd` runtime calls, so the compiled binary emits a Perfetto-viewable trace at exit. See [`docs/profiling.md`](profiling.md). |
| 2 | `PiccelerFiltersToConvPass` (`src/picceler_filters_to_conv_pass.cpp`) | Rewrites `sharpen` / `box_blur` / `gaussian_blur` / `edge_detect` / `emboss` into a canonical `picceler.convolution` + kernel pair. Collapses five op-specific lowerings into one, so every later pass only has to know how to lower `convolution`. |

`PiccelerAddProfilingPass` must run in this exact spot: after the canonicalizer (so dead/folded ops
never show up mislabeled in the trace) and before `PiccelerFiltersToConvPass` (so a `gaussian_blur`
in the trace reads "gaussian_blur", not the "convolution" it gets rewritten into one pass later).

After this phase, the only "filter" op left in the IR is `picceler.convolution`.

`sharpen`/`box_blur`/`gaussian_blur` take two different paths through `PiccelerFiltersToConvPass`
depending on whether their strength/radius argument is a compile-time constant:

* **Constant** (the common case: a literal, or something the canonicalizer folded to one) — kernel
  weights are computed host-side (`calculateSharpenKernel`, `calculateBoxBlurKernel`,
  `calculateGaussianKernel`) into a `picceler.kernel.const` + `picceler.kernel<RxC>`, exactly as
  before. `PiccelerKernelToMemrefPass` (Phase 3) later materializes that into a fixed-size
  `memref.alloca`.
* **Runtime** (e.g. the argument traces back to a function parameter) — there is no compile-time
  value to size a `picceler.kernel<RxC>` with, so `buildSharpenKernelDynamic` /
  `buildBoxBlurKernelDynamic` / `buildGaussianKernelDynamic` build the *same* arithmetic directly as
  arith/memref/affine (and, for gaussian's `exp` term, `math`) ops, writing straight into a memref
  that's fed to `picceler.convolution` as-is. `sharpen`'s kernel is always 3x3 regardless of
  strength, so it only needs a small fixed-size `memref.alloca`; `box_blur`/`gaussian_blur`'s kernel
  dimensions (`2*radius+1`) are themselves runtime values, so those get a dynamically-shaped
  `memref<?x?xf64>` from `memref.alloc` instead (see the "Note" at the end of Phase 4 for why
  `alloc` and not `alloca` here). Either way, `picceler.convolution`'s kernel operand was already
  typed `Picceler_AnyKernelType = AnyTypeOf<[Picceler_KernelType, AnyMemRef]>` (`tablegen/ops.td`),
  so no dialect/op changes were needed to accept it — `getKernelNeighborhoodSize`
  (`src/ops/convolution.cpp`) just needed to read a dynamic memref's shape with `memref.dim` instead
  of assuming compile-time-known dimensions, the same thing `dilate`/`erode`'s own
  `getNeighborhoodSize` already did for their radius (see Phase 3 below).

This two-path design — rather than *always* building the kernel dynamically, even for a constant
radius — was a deliberate choice: it keeps the constant case's IR exactly as compact and
canonicalizer/test-friendly as before (a single dense-attribute kernel, still foldable by
`IdentityConvolutionPattern`), and only pays for runtime kernel-fill loops when the input genuinely
requires them.

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
  doesn't fit the generic elementwise/neighborhood shapes). A compile-time-constant angle is
  validated and normalized host-side, same as always; a runtime angle gets the same validation
  (multiple of 90) and normalization (`((a % 360) + 360) % 360`) rebuilt out of `arith.remsi` /
  `arith.cmpi` instead, with an invalid angle triggering a `func.call @abort` guarded by `scf.if`
  (the same "runtime guard, not a compile error" shape `ElementWiseBinaryOpToAffine`'s dimension
  check below already uses) rather than failing to compile.
* `NeighbourhoodOpsToAffine` — ops with `NeighbourhoodOpInterface`: `convolution` (how
  `sharpen`/`box_blur`/`gaussian_blur`/`edge_detect`/`emboss` reach this pass, per Phase 1),
  `dilate`, `erode`. Emits an `scf.if` per sampled neighbor to guard out-of-bounds reads at the
  image border. Each op's `getNeighborhoodSize()` returns the taps-per-row/column as an
  `affine.parallel` bound — `dilate`/`erode` compute it from their radius operand (`2*radius+1`)
  regardless of whether that's a constant or runtime value; `convolution` reads it from its kernel
  operand's shape, either the compile-time `picceler.kernel<RxC>`/`memref<RxCxf64>` case or, for a
  genuinely dynamically-shaped `memref<?x?xf64>` kernel (Phase 1's runtime `box_blur`/
  `gaussian_blur` path), with `memref.dim` instead.
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
| 6 | `createSCFToControlFlowPass` | Lowers the `scf.if` from Phase 3's bounds/dimension-mismatch/angle-validity guards, plus whatever step 4 introduced, into unstructured `cf` branches. Must come after step 4, which is what produces most of the `scf` ops. |
| 7 | `createArithToLLVMConversionPass`, `createConvertMathToLLVMPass`, `createConvertControlFlowToLLVMPass`, `createConvertFuncToLLVMPass` | Lower `arith`, `math`, `cf`, and any remaining `func` ops to LLVM. `func` ops are normally already gone by step 1, so the `func`-to-LLVM pass here is a no-op safety net. The `math`-to-LLVM pass is what makes `math.exp` (gaussian_blur's runtime kernel path, Phase 1) and `sqrt()`/`pow()` translatable at all — without it, any `math` op that survives constant folding fails at LLVM IR translation. |
| 8 | `createReconcileUnrealizedCastsPass` (2nd) | Final cleanup — steps 4–7 each introduce their own casts converting operand/result types piecemeal. After this the IR should be fully LLVM dialect, ready for translation to LLVM IR. |

**Note:** most memref buffers in this pipeline are `memref.alloca` (kernel buffers, Phase 3/Phase 1)
rather than `memref.alloc`, and image buffers are built directly around a pointer the C runtime
already owns (Phase 2's `buildImageMemref`) — so there's no buffer-deallocation pass anywhere in
this pipeline. The two exceptions are also the two places a buffer's size can't be bounded at
compile time: `RotateToAffine`/`NeighbourhoodOpsToAffine`/`ElementWiseBinaryOpToAffine`/
`ElementWiseUnaryOpToAffine`/`CropToAffine`'s *output* image buffers (Phase 3), and
`buildBoxBlurKernelDynamic`/`buildGaussianKernelDynamic`'s runtime-sized kernel buffer for a
non-constant `box_blur`/`gaussian_blur` radius (Phase 1) — both intentionally use `memref.alloc`
instead, to avoid an unbounded stack allocation. These still never get an explicit `memref.dealloc`;
the leak is a known, pre-existing limitation of this pipeline, not something introduced here.

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

* Rewrites one high-level Picceler op into another → `addHighLevelOptimizationPasses()` (see
  `PiccelerFiltersToConvPass` for the pattern to follow, including how it branches between a
  host-computed constant kernel and one built from runtime arith/memref/affine ops depending on
  whether its input is a compile-time constant).
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
