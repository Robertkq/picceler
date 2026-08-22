This document aims to provide information about the picceler language, more specifically it's syntax and builtin functionalities.

# Syntax

### Comments

`#` starts a comment that runs to the end of the line. There is no block-comment syntax.

```
# This whole line is a comment
img = load_image("~/Pictures/cat.png") # so is everything after this '#'
```

### Program structure

A picceler source file is a sequence of top-level statements. Two shapes are accepted:

* One or more function definitions, one of which is named `main` — `main` is the entry point.
* A sequence of statements with **no** `main` function — the compiler implicitly wraps them in a
  generated `main` for you (any other function definitions in the file are left as-is, only the
  non-function statements are moved into the generated `main`). This is why `examples/photo_pipeline.pic`
  and `examples/blend_and_compare.pic` can be flat scripts with no `def main()` at all.

Mixing the two — defining `main` explicitly **and** having other top-level statements outside of any
function — is a compile error.

```
# examples/language_tour.pic (excerpt) — explicit main, no implicit wrapping needed
def classify(value: f64) {
    if (value < 10) {
        print("  {} is a small number\n", value)
    }
}

def main() {
    print("-- picceler language tour --\n")
    ...
}
```

```
# examples/photo_pipeline.pic (excerpt) — no 'main' defined, gets implicitly wrapped
photo = load_image("../img/cat.png")
show_image(photo)
brightened = brightness(photo, 15)
```

### Variables & assignment

Variables are dynamically typed and require no declaration keyword — an assignment is what
introduces a name:

```
name = expr
```

```
pathx = "~/Pictures/cat.png"
img = load_image(pathx)
```

### Literals

* **Numbers** — `f64` floating point values, with an optional leading `-` directly against the
  first digit (e.g. `-1`, `3.14`, `-0.5`). There is currently no separate integer literal syntax.
* **Strings** — double-quoted, e.g. `"cat.png"`. Recognized escape sequences: `\n` (newline),
  `\t` (tab), `\r` (carriage return), `\\` (backslash), `\0` (null byte), plus escaped single and
  double quote characters. A leading `~` in a string literal is expanded to the user's home
  directory at parse time, so `"~/Pictures/cat.png"` works the same as it would in a shell.
* **Kernels** — `N`x`M` matrices of numbers, every row must be the same length:
  `[[1,2,3],[4,5,6],[7,8,9]]`.

### Type annotations

`int64`, `f64`, `string`, and `image` are the recognized type names. They only appear in function
parameter and return-type positions (see "Function definitions & calls" below) — variables
themselves are never annotated.

### Operators & expression precedence

From lowest to highest precedence:

1. Relational: `==`, `!=`, `<`, `>`, `<=`, `>=`
2. Additive: `+`, `-`
3. Multiplicative: `*`, `/`

Parentheses `( expr )` can be used to override precedence.

```
root_val = sqrt(25.0)
cubed = pow(2.0, 3.0)
complex_calc = -1 + 2 * 3 + root_val + cubed
```

**Current limitations:**
* There are no logical operators (`&&`, `||`, `!`) — conditions are built from a single relational
  expression.
* There is no unary negation operator for arbitrary expressions. `-1` is a negative *number
  literal* (the `-` must be directly followed by a digit), but `-x` or `-(a + b)` are not valid —
  write `0 - x` instead.
* `sqrt()`/`pow()` only work when every argument is a compile-time-constant literal (as in the
  snippet above). There is currently no lowering pass from the `math` dialect to LLVM, so the only
  reason these calls compile at all is that the canonicalizer constant-folds them away entirely
  before that becomes a problem — calling either with a variable, function parameter, or any other
  runtime-computed value fails to compile with `missing LLVMTranslationDialectInterface registration
  for dialect for op: math.sqrt` (or `math.powf`).

### Function definitions & calls

```
def name(param1: type1, param2: type2, ...) -> returnType {
    ...
    return expr
}
```

The parameter list and `-> returnType` are both optional (a function with no declared return type
returns nothing). Calling a function uses the familiar `name(arg1, arg2, ...)` syntax.

```
# examples/language_tour.pic (excerpt) — a function with a typed parameter,
# and calling it
def classify(value: f64) {
    if (value < 10) {
        print("  {} is a small number\n", value)
    } else if (value < 20) {
        print("  {} is a medium number\n", value)
    } else {
        print("  {} is a large number\n", value)
    }
}

def main() {
    classify(3.0)
}
```

### Control flow

`if (cond) { ... }`, with optional `else { ... }` or chained `else if (cond) { ... }`:

```
if (complex_calc == 18){
    print("Binary operations passed successfully! \n")
} else if (complex_calc == 0) {
    print("Got zero\n")
} else {
    print("Something else\n")
}
```

### For loops

```
for (var = lowerBound .. upperBound [step stepExpr]) {
    ...
}
```

`step` defaults to `1` when omitted.

```
for (i = 1 .. 5) {
    print("Binary operations passed successfully! \n")
}
```

# Functionality

## Builtin Functions

* **load_image(string)** -> loads the image specified by `string` and assigns it to a variable
* **save_image(img, string)** -> saves `img` to the path specified by `string`
* **show_image(img)** -> opens a window to show `img`
* **read_number(string)** -> prompts with `string` and reads a number (`f64`) from the keyboard
* **read_string(string)** -> prompts with `string` and reads a string from the keyboard
* **print(string, ...)** -> prints `string` to the console, substituting each `{}` placeholder in
  order with the remaining arguments, e.g. `print("x = {}, y = {}\n", x, y)`
* **sqrt(f64)** -> square root of the argument. Argument must be a compile-time constant — see
  "Operators & expression precedence" above.
* **pow(f64, f64)** -> the first argument raised to the power of the second argument. Both
  arguments must be compile-time constants — see "Operators & expression precedence" above.

## Builtin Operations

* **brightness(img, value)** -> Eltwise addition of `value` to every pixel of `img`
* **invert(img)** -> Eltwise inversion of every pixel of `img`, following the `255 - pixel` formula.
* **convolution(img, kernel)** -> Performs a convolution on `img` using the specified `kernel`
* **sharpen(img, value)** -> Adjusts the contrast between adjacent pixels to increase the sharpness of `img`
* **box_blur(img, radius)** -> Applies a simple box blur to `img` by averaging pixels within the given `radius`
* **gaussian_blur(img, radius)** -> Applies a smooth, weighted gaussian blur to `img` based on the specified `radius`
* **edge_detect(img)** -> Detects and highlights the outlines and edges within `img`
* **emboss(img)** -> Applies a 3D effect to `img` by highlighting pixel intensity differences
* **rotate(img, angle)** -> Rotates `img` by `angle` degrees (current implementation supports multiples of 90; positive
  is counter-clockwise, negative is clockwise). For 90/270 rotations, output dimensions are swapped to fit the rotated
  image so the full content is preserved (no clipping)
* **crop(img, x, y, width, height)** -> Crops `img` to the `width`x`height` rectangle whose top-left corner is at (`x`, `y`)
* **dilate(img, radius)** -> Expands bright regions of `img` using a `radius`-sized neighborhood, so foreground areas grow
* **erode(img, radius)** -> Shrinks bright regions of `img` using a `radius`-sized neighborhood, so foreground areas contract
* **diff(img1, img2)** -> Computes the pixel-wise difference between `img1` and `img2`
* **blend(img1, img2, weight)** -> Blends `img1` and `img2` together using `weight` as the mix factor

## Builtin Types

* **string** -> Anything that is inside quotes e.g `"cat.png"`
* **kernel** -> `N`x`M` matrices, e.g `kernel = [ [1,2,3], [4,5,6], [7,8,9] ]`
* **image** -> Image data types, translated to a struct `Image { i32, i32, ptr }`


# Examples

The [examples](./examples/) directory has four small showcase programs, each demonstrating a
different style of picceler program:

* **photo_pipeline.pic** — chains several image effects together over a real photo and saves the
  results.
* **blend_and_compare.pic** — composites two images together with `blend()` and `diff()`.
* **language_tour.pic** — functions, `if`/`else if`/`else`, `for` loops, and `print()` formatting,
  with no image operations.
* **interactive_dilate.pic** — the `read_string()`/`read_number()` interactive-CLI pattern.

Run any of them from your build directory (see [BUILD.md](BUILD.md)):

```
./picceler -o myExecutable ./examples/FILENAME.pic
./myExecutable # Try running it!
```

