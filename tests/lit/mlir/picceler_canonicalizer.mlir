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

