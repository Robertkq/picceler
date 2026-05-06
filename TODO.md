# TODO

## Done

- [x] Expand `~` to the home directory.
- [x] Create the `picceler-opt` tool.

## In progress

- [~] Add MLIR LIT coverage for each pass using `picceler-opt`.
	- [x] Basic `picceler-opt` test setup exists.
	- [ ] Add one focused test file per pass.
	- [ ] Cover success cases for every lowering path.
	- [ ] Add failure/diagnostic cases for invalid input where useful.

## Next

### 1. Operations

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

### 2. Custom for-loops

- [ ] Decide whether this is syntax sugar or a standard-library style feature.
- [ ] Add `get_pixel(img, x, y)` support.
- [ ] Keep `img.get_pixel(x, y)` syntax working.
- [ ] Add matching `set_pixel(...)` or other image access helpers if needed.
- [ ] Decide whether this should enable a small standard library.

### 3. Custom functions for users

- [ ] Define the user-facing function syntax.
- [ ] Decide how functions are stored in the AST.
- [ ] Lower function calls through MLIR or the existing compilation pipeline.
- [ ] Add parser and semantic tests.

### 4. Better docs

- [ ] Add a `tests/README.md` that explains how to run and add lit tests.
- [ ] Add a `PASSES.md` or `PIPELINE.md` that explains the compiler pass order and responsibilities.
- [ ] Add a clearer overview to the main Doxygen page.
- [ ] Pull key content from `LANGUAGE.md` into the docs.
- [ ] Expand `LANGUAGE.md` with syntax, examples, and edge cases.
- [ ] Expand `BUILD.md` with build options, test options, and install notes.
- [ ] Add examples for the language and built-in operations.

### 5. Packaging and installs

- [ ] Produce generic Linux binaries.
- [ ] Decide whether Windows packages are worth supporting now.
- [ ] Add an install target for users building from source.
- [ ] Document the expected install and run workflow.