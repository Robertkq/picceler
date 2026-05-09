# TODO

## Done

- [x] Expand `~` to the home directory.
- [x] Create the `picceler-opt` tool.
- [x] Add MLIR LIT coverage for the current pass-level lowering flow.

## Current focus

The compiler currently has a small set of lowering passes that are worth keeping stable.
Prefer tests that prove the rewrite happened without overfitting to details that are likely to change.

### 1. Keep the working pass surface healthy

- [ ] Keep one focused lit file per maintained pass.
- [ ] Prefer structural checks over exact full-IR snapshots for unstable lowerings.
- [ ] Add diagnostics only where the pass is expected to reject invalid input.

#### Currently maintained

- [x] `picceler-filters-to-conv`
- [x] `picceler-kernel-to-memref`
- [ ] `picceler-to-affine`
- [ ] `picceler-to-llvmir`

#### Low-maintenance / best-effort

- [ ] Keep affine lowering working, but avoid making the test suite depend on fragile IR details.
- [ ] Keep LLVM IR lowering working, but keep expectations broad and update them only when the backend contract changes.
- [ ] Do not add broad feature support to backend passes until the lowering pipeline is more stable.

## Next

### 2. Operations

Break each operation into the same implementation slices:

- parser and AST support
- MLIR op / lowering support
- pass integration
- tests

#### Geometric

- [ ] `resize()` - scale an image up or down to a target width and height, ideally with a choice of interpolation later.
- [ ] `rotate()` - rotate an image around its center by an angle, with sensible handling for empty corners and bounds.
- [ ] `crop()` - extract a rectangular region from an image using x/y offsets and width/height.

#### Morphological

- [ ] `dilate()` - expand bright regions using a neighborhood kernel so foreground areas grow.
- [ ] `erode()` - shrink bright regions using a neighborhood kernel so foreground areas contract.

#### Dual-image operations

- [ ] `blend()` - combine two images into one using an alpha or mix factor.
- [ ] `diff()` - compare two images pixel-by-pixel and produce a difference image or mask.

### 3. Custom for-loops

- [ ] Decide whether this is syntax sugar or a standard-library style feature.
- [ ] Add `get_pixel(img, x, y)` support.
- [ ] Keep `img.get_pixel(x, y)` syntax working.
- [ ] Add matching `set_pixel(...)` or other image access helpers if needed.
- [ ] Decide whether this should enable a small standard library.

### 4. Custom functions for users

- [ ] Define the user-facing function syntax.
- [ ] Decide how functions are stored in the AST.
- [ ] Lower function calls through MLIR or the existing compilation pipeline.
- [ ] Add parser and semantic tests.

### 5. Better docs

- [ ] Add a `tests/README.md` that explains how to run and add lit tests.
- [ ] Add a `PASSES.md` or `PIPELINE.md` that explains the compiler pass order and responsibilities.
- [ ] Add a clearer overview to the main Doxygen page.
- [ ] Pull key content from `LANGUAGE.md` into the docs.
- [ ] Expand `LANGUAGE.md` with syntax, examples, and edge cases.
- [ ] Expand `BUILD.md` with build options, test options, and install notes.
- [ ] Add examples for the language and built-in operations.

### 6. Packaging and installs

- [ ] Produce generic Linux binaries.
- [ ] Decide whether Windows packages are worth supporting now.
- [ ] Add an install target for users building from source.
- [ ] Document the expected install and run workflow.