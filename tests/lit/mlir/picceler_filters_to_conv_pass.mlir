// RUN: %picceler-opt --picceler-filters-to-conv -split-input-file %s | FileCheck %s

func.func @SharpenImage(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 10 : i64} : () -> i64
    %0 = "picceler.sharpen" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @SharpenImage(%arg0: !picceler.image) -> !picceler.image
// CHECK: %[[KERNEL:.*]] = "picceler.kernel.const"() 
// CHECK-LITERAL: <{values = dense<[[0.000000e+00, -4.000000e-01, 0.000000e+00], [-4.000000e-01, 2.600000e+00, -4.000000e-01], [0.000000e+00, -4.000000e-01, 0.000000e+00]]> : tensor<3x3xf64>}>
// CHECK: : () -> !picceler.kernel<3 x 3>
// CHECK-NEXT: %[[CONV:.*]] = "picceler.convolution"(%arg0, %[[KERNEL]]) : (!picceler.image, !picceler.kernel<3 x 3>) -> !picceler.image
// CHECK-NEXT: return %[[CONV]] : !picceler.image

// -----

func.func @BoxBlurImage(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 2 : i64} : () -> i64
    %0 = "picceler.box_blur" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @BoxBlurImage(%arg0: !picceler.image) -> !picceler.image
// CHECK: %[[KERNEL:.*]] = "picceler.kernel.const"()
// CHECK-LITERAL: <{values = dense<[4.000000e-02]> : tensor<3x3xf64>}>
// CHECK: : () -> !picceler.kernel<5 x 5>
// CHECK-NEXT: %[[CONV:.*]] = "picceler.convolution"(%arg0, %[[KERNEL]]) : (!picceler.image, !picceler.kernel<5 x 5>) -> !picceler.image
// CHECK-NEXT: return %[[CONV]] : !picceler.image

// -----

func.func @GaussianBlurImage(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 2 : i64} : () -> i64
    %0 = "picceler.gaussian_blur" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @GaussianBlurImage(%arg0: !picceler.image) -> !picceler.image
// CHECK: %[[KERNEL:.*]] = "picceler.kernel.const"() <{values = dense<
// CHECK: > : tensor<5x5xf64>}> : () -> !picceler.kernel<5 x 5>
// CHECK-NEXT: %[[CONV:.*]] = "picceler.convolution"(%arg0, %[[KERNEL]]) : (!picceler.image, !picceler.kernel<5 x 5>) -> !picceler.image
// CHECK-NEXT: return %[[CONV]] : !picceler.image

// -----

func.func @EdgeDetectImage(%arg0 : !picceler.image) -> !picceler.image {
    %0 = "picceler.edge_detect" (%arg0) : (!picceler.image) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @EdgeDetectImage(%arg0: !picceler.image) -> !picceler.image
// CHECK: %[[KERNEL:.*]] = "picceler.kernel.const"()
// CHECK-LITERAL: <{values = dense<[[-1.000000e+00, -1.000000e+00, -1.000000e+00], [-1.000000e+00, 8.000000e+00, -1.000000e+00], [-1.000000e+00, -1.000000e+00, -1.000000e+00]]> : tensor<3x3xf64>}>
// CHECK: : () -> !picceler.kernel<3 x 3>
// CHECK-NEXT: %[[CONV:.*]] = "picceler.convolution"(%arg0, %[[KERNEL]]) : (!picceler.image, !picceler.kernel<3 x 3>) -> !picceler.image
// CHECK-NEXT: return %[[CONV]] : !picceler.image

// -----

func.func @EmbossImage(%arg0 : !picceler.image) -> !picceler.image {
    %0 = "picceler.emboss" (%arg0) : (!picceler.image) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @EmbossImage(%arg0: !picceler.image) -> !picceler.image
// CHECK: %[[KERNEL:.*]] = "picceler.kernel.const"()
// CHECK-LITERAL: <{values = dense<[[-2.000000e+00, -1.000000e+00, 0.000000e+00], [-1.000000e+00, 1.000000e+00, 1.000000e+00], [0.000000e+00, 1.000000e+00, 2.000000e+00]]> : tensor<3x3xf64>}>
// CHECK: : () -> !picceler.kernel<3 x 3>
// CHECK-NEXT: %[[CONV:.*]] = "picceler.convolution"(%arg0, %[[KERNEL]]) : (!picceler.image, !picceler.kernel<3 x 3>) -> !picceler.image
// CHECK-NEXT: return %[[CONV]] : !picceler.image

// -----

// Below: sharpen/box_blur/gaussian_blur with a *non-constant* strength/radius (e.g. traced back to
// a function parameter, matching what mlir_gen.cpp's coerceValueToInt64 emits for a runtime f64
// argument -- arith.fptosi, not arith.constant). Unlike the compile-time cases above, these can't
// produce a picceler.kernel.const (its values attribute must be a compile-time dense attribute), so
// they build the kernel directly into a memref with arith/memref/affine/math ops instead -- see
// buildSharpenKernelDynamic/buildBoxBlurKernelDynamic/buildGaussianKernelDynamic in
// picceler_filters_to_conv_pass.cpp. picceler.convolution accepts that memref kernel operand
// natively (Picceler_AnyKernelType already allows AnyMemRef).
//
// This block intentionally is not given its own split marker like the ones above: MLIRContext only
// loads a dialect the first time it is referenced by name, either while parsing this file's text or
// while an op is built programmatically ahead of time -- and unlike the constant-kernel path above,
// nothing in a legitimate sharpen/box_blur/gaussian_blur *input* mentions arith/memref/affine/math
// textually, so this throwaway function loads them upfront for the whole block (a split marker
// would reset that per test function, since -split-input-file gives each split its own fresh
// context).
func.func private @__load_dialects_for_runtime_kernel_tests() {
  %c0 = arith.constant 0 : index
  %f = arith.constant 0.0 : f64
  %e = math.exp %f : f64
  %m = memref.alloca() : memref<1xf64>
  memref.store %f, %m[%c0] : memref<1xf64>
  affine.for %i = 0 to 1 {
  }
  return
}

func.func @SharpenImageRuntime(%arg0 : !picceler.image, %strengthF64 : f64) -> !picceler.image {
    %value = arith.fptosi %strengthF64 : f64 to i64
    %0 = "picceler.sharpen" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @SharpenImageRuntime
// CHECK-NOT: "picceler.kernel.const"
// CHECK: memref.alloca() : memref<3x3xf64>
// CHECK-COUNT-9: memref.store
// CHECK: "picceler.convolution"(%arg0, %{{.*}}) : (!picceler.image, memref<3x3xf64>) -> !picceler.image
// CHECK-NOT: "picceler.sharpen"

func.func @BoxBlurImageRuntime(%arg0 : !picceler.image, %radiusF64 : f64) -> !picceler.image {
    %radius = arith.fptosi %radiusF64 : f64 to i64
    %0 = "picceler.box_blur" (%arg0, %radius) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @BoxBlurImageRuntime
// CHECK-NOT: "picceler.kernel.const"
// CHECK: arith.addi
// CHECK: memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf64>
// CHECK: affine.parallel
// CHECK: memref.store
// CHECK: "picceler.convolution"(%arg0, %{{.*}}) : (!picceler.image, memref<?x?xf64>) -> !picceler.image
// CHECK-NOT: "picceler.box_blur"

func.func @GaussianBlurImageRuntime(%arg0 : !picceler.image, %radiusF64 : f64) -> !picceler.image {
    %radius = arith.fptosi %radiusF64 : f64 to i64
    %0 = "picceler.gaussian_blur" (%arg0, %radius) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @GaussianBlurImageRuntime
// CHECK-NOT: "picceler.kernel.const"
// CHECK: memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?xf64>
// CHECK: affine.parallel{{.*}}reduce
// CHECK: math.exp
// CHECK: affine.parallel
// CHECK: memref.load
// CHECK: arith.divf
// CHECK: "picceler.convolution"(%arg0, %{{.*}}) : (!picceler.image, memref<?x?xf64>) -> !picceler.image
// CHECK-NOT: "picceler.gaussian_blur"