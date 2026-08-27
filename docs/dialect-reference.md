# Picceler MLIR Dialect Reference

This document summarizes the MLIR dialects that show up most often in Picceler, based on the current compiler pipeline and lowering passes.

The short version is:

- Picceler defines the source-level image operations and custom types.
- `arith`, `affine`, `func`, `memref`, `scf`, and `LLVM` are the main MLIR dialects used during lowering.
- `builtin`/core IR is used for the module container and general MLIR plumbing.

## 1. Picceler Dialect

File: `tablegen/ops.td`, `tablegen/types.td`, `tablegen/interfaces.td`

This is the custom dialect for the language. Most front-end code builds these ops directly, and the lowering passes then convert them to standard MLIR or LLVM dialects.

| Op | What it does | Expected inputs | Result |
| --- | --- | --- | --- |
| `picceler.load_image` | Loads an image from disk | `picceler.string` filename | `picceler.image` |
| `picceler.save_image` | Saves an image to disk | `picceler.image`, `picceler.string` filename | none |
| `picceler.show_image` | Displays an image in a window | `picceler.image` | none |
| `picceler.brightness` | Adds a signed brightness offset to pixels | `picceler.image`, `i64` value | `picceler.image` |
| `picceler.invert` | Inverts pixel values | `picceler.image` | `picceler.image` |
| `picceler.sharpen` | Sharpens an image | `picceler.image`, `i64` strength | `picceler.image` |
| `picceler.box_blur` | Box blur over a square neighborhood | `picceler.image`, `i64` radius | `picceler.image` |
| `picceler.gaussian_blur` | Gaussian blur over a square neighborhood | `picceler.image`, `i64` radius | `picceler.image` |
| `picceler.edge_detect` | Edge detection filter | `picceler.image` | `picceler.image` |
| `picceler.emboss` | Emboss filter | `picceler.image` | `picceler.image` |
| `picceler.rotate` | Rotates an image by a right-angle/normalized angle in the current lowering | `picceler.image`, `i64` angle | `picceler.image` |
| `picceler.convolution` | Applies a kernel-based neighborhood reduction | `picceler.image`, `picceler.kernel` or `memref` kernel | `picceler.image` |
| `picceler.erode` | Erosion over a square neighborhood | `picceler.image`, `i64` radius | `picceler.image` |
| `picceler.dilate` | Dilation over a square neighborhood | `picceler.image`, `i64` radius | `picceler.image` |
| `picceler.diff` | Pixel-wise image difference | `picceler.image`, `picceler.image` | `picceler.image` |
| `picceler.blend` | Blends two images with a weight | `picceler.image`, `picceler.image`, `float64` weight | `picceler.image` |
| `picceler.crop` | Crops a rectangular region | `picceler.image`, `i64 x`, `i64 y`, `i64 width`, `i64 height` | `picceler.image` |
| `picceler.read_string` | Reads a string from the keyboard | `picceler.string` prompt | `picceler.string` |
| `picceler.read_number` | Reads a number from the keyboard | `picceler.string` prompt | `float64` |
| `picceler.print` | Prints a value to the console | any single MLIR value | none |
| `picceler.string.const` | Produces a constant string value | `StrAttr` value | `picceler.string` |
| `picceler.kernel.const` | Produces a constant kernel value | `F64ElementsAttr` values | `picceler.kernel` |

### Custom Picceler Types

| Type | Meaning |
| --- | --- |
| `picceler.image` | Abstract image value used throughout the high-level pipeline |
| `picceler.string` | Runtime string value, usually prompts or filenames |
| `picceler.kernel<rowsxcols>` | Constant 2D kernel used by convolution-style ops |

## 2. `arith` Dialect

File(s): `src/mlir_gen.cpp`, `src/picceler_to_affine_pass.cpp`, `src/ops/*.cpp`

This is the most common utility dialect in the project. It handles constants, casts, comparisons, and numeric math.

| Op | What it does | Expected inputs | Typical use here |
| --- | --- | --- | --- |
| `arith.constant` variants | Produces literal integers/floats/index values | attribute value only | Build pixel constants, loop bounds, radius constants |
| `arith.index_cast` | Casts between `index` and integer types | `index` or integer | Convert loop/index math to `i64` for LLVM GEPs |
| `arith.extui` / `arith.extsi` | Extends a smaller integer to a larger one | integer input | Expand pixel bytes before arithmetic |
| `arith.trunci` | Truncates a larger integer to a smaller one | integer input | Clamp back to `i8` or similar |
| `arith.ui_to_fp` | Converts unsigned integer to floating point | integer input | Convert byte pixels to `float64` for filters |
| `arith.fp_to_ui` | Converts floating point to unsigned integer | floating-point input | Convert clamped pixel math back to `i8` |
| `arith.fp_to_si` | Converts floating point to signed integer | floating-point input | Convert runtime `float64` values to `i64` when needed |
| `arith.addi` / `subi` / `muli` | Integer arithmetic | integer operands | Pixel offset math, neighborhood math |
| `arith.addf` / `subf` / `mulf` | Floating-point arithmetic | floating-point operands | Blend/convolution/dilate accumulators |
| `arith.cmpi` | Integer comparison | integer operands | Bounds checks and angle checks |
| `arith.maximumf` / `minimumf` | Floating-point min/max | floating-point operands | Clamp image accumulators |

## 3. `func` Dialect

File(s): `src/picceler_ops_to_func_calls_pass.cpp`, `src/picceler_to_llvm_ir_pass.cpp`, `src/mlir_gen.cpp`

This dialect is used for function boundaries and runtime library calls.

| Op | What it does | Expected inputs | Result |
| --- | --- | --- | --- |
| `func.func` | Declares or defines a function | function type | function symbol |
| `func.call` | Calls a function | operands matching callee signature | call results |
| `func.return` | Returns from a function | return operands matching signature | none |

Common runtime calls in this project include image I/O and keyboard input wrappers such as `piccelerLoadImage`, `piccelerSaveImage`, `piccelerShowImage`, `piccelerReadNumber`, and `piccelerReadString`.

## 4. `affine` Dialect

File(s): `src/picceler_to_affine_pass.cpp`

This dialect is used for the image-processing lowerings that become nested loops.

| Op | What it does | Expected inputs | Typical use here |
| --- | --- | --- | --- |
| `affine.for` | Compile-time affine loop | lower bound, upper bound, step | Iterate over image rows/cols and neighborhood windows |
| `affine.apply` | Computes affine expressions on loop/index values | affine map + operands | Translate row/col coordinates into linear addresses |

Important note: several neighborhood-based ops currently require a statically known neighborhood size. If a radius comes from runtime input, the current affine lowering may reject it or need a different lowering strategy.

## 5. `memref` Dialect

File(s): `src/picceler_kernel_to_memref_pass.cpp`, `src/picceler_to_affine_pass.cpp`, `src/picceler_filters_to_conv_pass.cpp`

This dialect is used when data is materialized in stack or heap-like buffer form.

| Op | What it does | Expected inputs | Typical use here |
| --- | --- | --- | --- |
| `memref.alloca` | Allocates stack memory for a memref | memref type and optional sizes | Temporary kernel storage, accumulators |
| `memref.load` | Loads from a memref | memref + indices | Read kernel weights |
| `memref.store` | Stores to a memref | value + memref + indices | Write kernel data or intermediate values |

## 6. `scf` Dialect

File(s): `src/picceler_to_affine_pass.cpp`

This is used for structured control flow that is easier to lower than explicit branches in some parts of the pipeline.

| Op | What it does | Expected inputs | Typical use here |
| --- | --- | --- | --- |
| `scf.if` | Conditional region-based branch | boolean condition | Guard out-of-bounds pixel samples |

## 7. `LLVM` Dialect

File(s): `src/picceler_to_llvm_ir_pass.cpp`, `src/picceler_to_affine_pass.cpp`, `src/image_access_helper.cpp`

This is the final low-level dialect used before emitting LLVM IR and object code.

| Op | What it does | Expected inputs | Typical use here |
| --- | --- | --- | --- |
| `llvm.global` | Declares a global buffer | type, linkage, initial value | String literals or runtime constants |
| `llvm.addressof` | Gets the address of a global | global symbol | Turn a global into a pointer |
| `llvm.gep` | Pointer arithmetic | base pointer + indices | Access image buffers and struct fields |
| `llvm.load` | Loads from a pointer | typed pointer | Read pixels, image metadata, struct fields |
| `llvm.store` | Stores to a pointer | value + typed pointer | Write pixels or struct fields |
| `llvm.constant` | Produces a low-level constant | attribute value | Emit pointer offsets and integer constants |

## 8. Practical Type Rules To Remember

- `picceler.read_number` returns `float64`, so if a downstream op expects `i64`, the frontend must insert an explicit float-to-int cast.
- `arith.index_cast` is only for `index` and integers; it is not a float cast.
- For `float64 -> i64`, use `arith.fp_to_si` in the signed case.
- Neighborhood-based ops like `dilate`, `erode`, and some blur/convolution paths still assume a radius or kernel size that the lowering can reason about.

## 9. Pipeline Snapshot

For the full pass-by-pass breakdown of how `picceler` dialect IR reaches `LLVM` dialect IR — phase
ordering, what each pass does, and why it's sequenced where it is — see
[`docs/compiler-internals.md`](compiler-internals.md).