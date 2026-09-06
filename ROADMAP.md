# Roadmap

Some features I'd like to add in the future:

## Buffer deallocation

`PiccelerToAffinePass` allocates the output buffer for every compute op with
`memref.alloc` (five sites, see `src/picceler_to_affine_pass.cpp`), and
nothing frees them, so a pipeline of N operations leaks N full resolution
buffers. A first attempt at wiring up deallocation did not work out and was
reverted; this needs revisiting with MLIR's bufferization deallocation
pipeline. See [Phase 4, Backend Lowering](docs/compiler-internals.md#6-phase-4--backend-lowering)
for how buffers are allocated today.

## GPU offloading

The plan is to add a function attribute to the picceler language marking a
function for GPU execution; functions carrying it would compile through
MLIR's `gpu` dialect and get offloaded, instead of lowering to CPU affine
loops. The existing lowering already emits `affine.parallel` over image rows
and columns with no cross iteration dependencies, which is the loop shape GPU
offload needs, so the loop structure is not the hard part. The hard parts are
moving buffers between host and device, and keeping images resident on the
device across a multi operation pipeline instead of copying back and forth
around every op. The SPIR-V backend (`GPUToSPIRV`) is the intended target
rather than NVVM, since SPIR-V is what SYCL and Level Zero consume.
