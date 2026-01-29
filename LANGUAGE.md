This document aims to provide information about the picceler language, more specifically it's syntax and builtin functionalities.

# Syntax

# Functionality

## Builtin Functions

* **load_image(string)** -> loads the image specified by `string` and assigns it to a variable
* **save_image(img, string)** -> saves `img` to the path specified by `string`
* **show_image(img)** -> opens a window to show `img`

## Builtin Operations

* **brightness(img, value)** -> Eltwise addition of `value` to every pixel of `img`
* **invert(img)** -> Eltwise inversion of every pixel of `img`, following the `255 - pixel` formula.
* **convolution(img, kernel)** -> Performs a convolution on `img` using the specified `kernel`
* **sharpen(img, value)** -> Adjusts the contrast between adjacent pixels to increase the sharpness of `img`
* **box_blur(img, radius)** -> Applies a simple box blur to `img` by averaging pixels within the given `radius`
* **gaussian_blur(img, radius)** -> Applies a smooth, weighted gaussian blur to `img` based on the specified `radius`
* **edge_detect(img)** -> Detects and highlights the outlines and edges within `img`
* **emboss(img)** -> Applies a 3D effect to `img` by highlighting pixel intensity differences

## Builtin Types

* **string** -> Anything that is inside quotes e.g `"cat.png"`
* **kernel** -> `N`x`M` matrices, e.g `kernel = [ [1,2,3], [4,5,6], [7,8,9] ]`
* **image** -> Image data types, translated to a struct `Image { i32, i32, ptr }`


# Examples

We have multiple picceler files that exemplify how to use the picceler language.

You can try compiling any of the following files from the [examples](./examples/) directory, by using:

```bash
./picceler -o myExecutable ./examples/<file>.pic
./myExecutable # Try running it!
```