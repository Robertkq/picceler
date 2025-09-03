# picceler

*picceler* (Pixel Accelerator) is a domain specific language for image processing.
The language aims to simlpify and fasten the development speed for image processing work.

# Compiler
## Frontend
* Should parse the DSL and generate a Abstract syntax tree
## Middleend
* MLIR Dialect
## Backend
* Lowering to LLVM IR & using SYCL to generate the kernel
* might not be able to use SYCL here due to complex procedure

# Language
## Syntax
* python-like
* immutable variables

## Builtin
* load_image
* save_image
* show_image
* blur()

## Functions (later)
* parameters
* return

