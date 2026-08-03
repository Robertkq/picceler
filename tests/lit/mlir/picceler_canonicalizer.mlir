// RUN: %picceler-opt --canonicalize -split-input-file %s | FileCheck %s

func.func @BrightnessFoldOnZero(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 0 : i64} : () -> i64
    %0 = "picceler.brightness" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @BrightnessFoldOnZero(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// -----

func.func @BrightnessChain(%arg0 : !picceler.image) -> !picceler.image {
    %value1 = "arith.constant"() {value = 10 : i64} : () -> i64
    %0 = "picceler.brightness" (%arg0, %value1) : (!picceler.image, i64) -> !picceler.image
    %value2 = "arith.constant"() {value = 20 : i64} : () -> i64
    %1 = "picceler.brightness" (%0, %value2) : (!picceler.image, i64) -> !picceler.image
    return %1 : !picceler.image
}

// CHECK-LABEL: func.func @BrightnessChain(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: %[[NEW_VALUE:.*]] = arith.constant 30 : i64
// CHECK-NEXT: %[[NEW_BRIGHTNESS:.*]] = "picceler.brightness"(%arg0, %[[NEW_VALUE]]) : (!picceler.image, i64) -> !picceler.image
// CHECK-NEXT: return %[[NEW_BRIGHTNESS]] : !picceler.image

// -----

func.func @BrightnessFoldZeroUses(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 20 : i64} : () -> i64
    %0 = "picceler.brightness" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %arg0 : !picceler.image
}

// CHECK-LABEL: func.func @BrightnessFoldZeroUses(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// ----- 

func.func @BrightnessChainResultsInZeroFold(%arg0 : !picceler.image) -> !picceler.image {
    %value1 = "arith.constant"() {value = 10 : i64} : () -> i64
    %0 = "picceler.brightness" (%arg0, %value1) : (!picceler.image, i64) -> !picceler.image
    %value2 = "arith.constant"() {value = -10 : i64} : () -> i64
    %1 = "picceler.brightness" (%0, %value2) : (!picceler.image, i64) -> !picceler.image
    return %1 : !picceler.image
}

// CHECK-LABEL: func.func @BrightnessChainResultsInZeroFold(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// -----

func.func @BrightnessNoFoldOnNonConstant(%arg0 : !picceler.image, %arg1 : i64) -> !picceler.image {
    %0 = "picceler.brightness" (%arg0, %arg1) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @BrightnessNoFoldOnNonConstant(%arg0: !picceler.image, %arg1: i64) -> !picceler.image
// CHECK-NEXT: %0 = "picceler.brightness"(%arg0, %arg1) : (!picceler.image, i64) -> !picceler.image
// CHECK-NEXT: return %0 : !picceler.image

// -----

func.func @InvertFoldOnChain(%arg0 : !picceler.image) -> !picceler.image {
    %0 = "picceler.invert" (%arg0) : (!picceler.image) -> !picceler.image
    %1 = "picceler.invert" (%0) : (!picceler.image) -> !picceler.image
    return %1 : !picceler.image
}

// CHECK-LABEL: func.func @InvertFoldOnChain(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// -----

func.func @SharpenFoldOnZero(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 0 : i64} : () -> i64
    %0 = "picceler.sharpen" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @SharpenFoldOnZero(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// -----

func.func @BoxBlurFoldOnZero(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 0 : i64} : () -> i64
    %0 = "picceler.box_blur" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @BoxBlurFoldOnZero(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// -----

func.func @GaussianBlurFoldOnZero(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 0 : i64} : () -> i64
    %0 = "picceler.gaussian_blur" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @GaussianBlurFoldOnZero(%arg0: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %arg0 : !picceler.image

// -----

func.func @EdgeDetectBypassSingleUseInvert(%arg0 : !picceler.image) -> !picceler.image {
    %0 = "picceler.invert" (%arg0) : (!picceler.image) -> !picceler.image
    %1 = "picceler.edge_detect" (%0) : (!picceler.image) -> !picceler.image
    return %1 : !picceler.image
}

// CHECK-LABEL: func.func @EdgeDetectBypassSingleUseInvert
// CHECK-SAME: (%[[INPUT:.*]]: !picceler.image) -> !picceler.image
// CHECK-NEXT: %[[EDGE:.*]] = "picceler.edge_detect"(%[[INPUT]]) : (!picceler.image) -> !picceler.image
// CHECK-NEXT: return %[[EDGE]] : !picceler.image

// -----

func.func @EdgeDetectBypassMultipleUseInvert(%arg0 : !picceler.image) -> !picceler.image {
    %alpha = arith.constant 0.5 : f64
    %0 = "picceler.invert" (%arg0) : (!picceler.image) -> !picceler.image
    %1 = "picceler.edge_detect" (%0) : (!picceler.image) -> !picceler.image
    %2 = "picceler.blend" (%0, %1, %alpha) : (!picceler.image, !picceler.image, f64) -> !picceler.image
    return %2 : !picceler.image
}

// CHECK-LABEL: func.func @EdgeDetectBypassMultipleUseInvert
// CHECK-SAME: (%[[INPUT:.*]]: !picceler.image) -> !picceler.image
// CHECK-DAG: %[[ALPHA:.*]] = arith.constant 5.000000e-01 : f64
// CHECK-DAG: %[[INVERT:.*]] = "picceler.invert"(%[[INPUT]]) : (!picceler.image) -> !picceler.image
// CHECK-DAG: %[[EDGE:.*]] = "picceler.edge_detect"(%[[INPUT]]) : (!picceler.image) -> !picceler.image
// CHECK: %[[BLEND:.*]] = "picceler.blend"(%[[INVERT]], %[[EDGE]], %[[ALPHA]]) : (!picceler.image, !picceler.image, f64) -> !picceler.image
// CHECK-NEXT: return %[[BLEND]] : !picceler.image

// -----

func.func @RotateFoldOnZero(%arg0 : !picceler.image) -> !picceler.image {
    %value = "arith.constant"() {value = 0 : i64} : () -> i64
    %0 = "picceler.rotate" (%arg0, %value) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @RotateFoldOnZero
// CHECK-SAME: (%[[INPUT:.*]]: !picceler.image) -> !picceler.image
// CHECK-NEXT: return %[[INPUT]] : !picceler.image

// -----

func.func @ChainedRotatesFolding(%arg0 : !picceler.image) -> !picceler.image {
    %value1 = "arith.constant"() {value = 90 : i64} : () -> i64
    %0 = "picceler.rotate" (%arg0, %value1) : (!picceler.image, i64) -> !picceler.image
    %value2 = "arith.constant"() {value = 180 : i64} : () -> i64
    %1 = "picceler.rotate" (%0, %value2) : (!picceler.image, i64) -> !picceler.image
    return %1 : !picceler.image
}

// CHECK-LABEL: func.func @ChainedRotatesFolding
// CHECK-SAME: (%[[INPUT:.*]]: !picceler.image) -> !picceler.image
// CHECK-DAG: %[[NEW_VALUE:.*]] = arith.constant 270 : i64
// CHECK: %[[NEW_ROTATE:.*]] = "picceler.rotate"(%[[INPUT]], %[[NEW_VALUE]]) : (!picceler.image, i64) -> !picceler.image
// CHECK-NEXT: return %[[NEW_ROTATE]] : !picceler.image

// -----

func.func @FoldIdentityConvolution(%arg0 : !picceler.image) -> !picceler.image {
    %kernel = "picceler.kernel.const"() <{values = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00],
     [0.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<3x3xf64>}> : () -> !picceler.kernel<3 x 3>
    %0 = "picceler.convolution" (%arg0, %kernel) : (!picceler.image, !picceler.kernel<3 x 3>) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @FoldIdentityConvolution
// CHECK-SAME: (%[[INPUT:.*]]: !picceler.image) -> !picceler.image
// CHECK: return %[[INPUT]] : !picceler.image

// -----

 func.func @FoldUnusedKernel(%arg0 : i64) -> i64 {
    %0 = "picceler.kernel.const"() <{values = dense<1.000000e+00> : tensor<1x1xf64>}> : () -> !picceler.kernel<1 x 1>
    return %arg0 : i64
 }

 // CHECK-LABEL: func.func @FoldUnusedKernel
 // CHECK-SAME: (%[[ARG0:.*]]: i64) -> i64
 // CHECK-NEXT: return %[[ARG0]] : i64


