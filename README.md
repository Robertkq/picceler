# picceler

*picceler* (Pixel Accelerator) is a compiled domain specific language for image processing.
The language aims to simlpify and fasten the development speed for image processing work (currently only supporting CPU x86 arch).

Picceler doesn't aim to be a production language, rather it aims to provide a good example of what you can achieve with MLIR.

# How to build & install

The process of building and installing can be quite long, please refer to the [How to build](BUILD.md) guide for this information.

# Language

The picceler language currently follows a typeless, immutable, python-like syntax. Keeping things simple. 

Please refer to the [Language](LANGUAGE.md) document for more precise information on syntax and builtin operations.

# Inner workings

This section will provide details about how, using MLIR, we can get from high level source code, to an intermediate representation (IR) and then to assembly for a specific platform.

WIP -- md document or latex would be cool


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

