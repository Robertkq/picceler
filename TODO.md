# TODO:

## 1. Operations:

### Geometric:
*   resize()
*   rotate()
*   crop()
### Morphological:
*    dilate()
*    erode()
### Dual-Image Operations
*    blend()
*    diff() 

## 2. custom for loops
* this also needs get_pixel(img, x, y), while keeing `img.get_pixel(x,y)` syntax
* probably more `get/set` functions for images
* this would also allow a standard library
* unlikely to implement this but would be cool

## 3. custom functions for users  

## 4. Expand `~` to home directory
* `~` doesnt expend automatically to home directory

## 5. Have MLIR LIT tests for each pass
* using the picceler-opt tool

## 6. Better docs
* Would be really nice to add stuff to the main page of picceler doxygen 
* Would be really nice to use LANGUAGE.md and/or other docs from repo in doxygen

## 7. Have some generic linux (maybe windows too) packages with binaries
* would be cool to run out of the box, 
* also implement install target for users that want to build manually