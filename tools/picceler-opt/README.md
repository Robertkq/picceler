## picceler-opt

This is the source for a custom mlir-opt executable that knows about picceler dialect and it's passes. It is used for LIT MLIR testing of passes. This executable allows running picceler and any other registered passes on input IR, then we use FileCheck to validate the output.