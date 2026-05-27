// RUN: %picceler-opt --picceler-to-affine -split-input-file %s | FileCheck %s

func.func @RotateImage(%arg0 : !picceler.image) -> !picceler.image {
    %angle = "arith.constant"() {value = 90 : i64} : () -> i64
    %0 = "picceler.rotate" (%arg0, %angle) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @RotateImage
// CHECK: call @piccelerCreateImage
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.select
// CHECK-NOT: "picceler.rotate"
// CHECK: return

// -----

func.func @DiffImages(%arg0 : !picceler.image, %arg1 : !picceler.image) -> !picceler.image {
    %0 = "picceler.diff" (%arg0, %arg1) : (!picceler.image, !picceler.image) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @DiffImages
// CHECK: func.call @abort
// CHECK: call @piccelerCreateImage
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.extui
// CHECK: arith.subi
// CHECK: arith.select
// CHECK-NOT: "picceler.diff"       
// CHECK: return

// -----

func.func @BlendImages(%arg0 : !picceler.image, %arg1 : !picceler.image) -> !picceler.image {
    %weight = "arith.constant"() {value = 0.5 : f64} : () -> f64
    %0 = "picceler.blend" (%arg0, %arg1, %weight) : (!picceler.image, !picceler.image, f64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @BlendImages
// CHECK: call @piccelerCreateImage
// CHECK: affine.for
// CHECK: affine.for
// CHECK: arith.uitofp
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.fptoui
// CHECK-NOT: "picceler.blend"
// CHECK: return

// -----

func.func @DilateImage(%arg0 : !picceler.image) -> !picceler.image {
    %radius = "arith.constant"() {value = 1 : i64} : () -> i64
    %0 = "picceler.dilate" (%arg0, %radius) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @DilateImage
// CHECK: call @piccelerCreateImage
// CHECK-COUNT-4: affine.for
// CHECK: arith.maximumf
// CHECK-NOT: "picceler.dilate"
// CHECK: return

// -----

func.func @ErodeImage(%arg0 : !picceler.image) -> !picceler.image {
    %radius = "arith.constant"() {value = 1 : i64} : () -> i64
    %0 = "picceler.erode" (%arg0, %radius) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @ErodeImage
// CHECK: call @piccelerCreateImage
// CHECK-COUNT-4: affine.for
// CHECK: arith.minimumf
// CHECK-NOT: "picceler.erode"
// CHECK: return

// -----

func.func @ConvolutionImage(%arg0 : !picceler.image) -> !picceler.image {
    %c0 = "arith.constant"() {value = 0 : index} : () -> index
    %c1 = "arith.constant"() {value = 1 : index} : () -> index
    %c0f = "arith.constant"() {value = 0.0 : f64} : () -> f64
    %kernel = memref.alloca() : memref<3x3xf64>
    memref.store %c0f, %kernel[%c0, %c0] : memref<3x3xf64>
    memref.store %c0f, %kernel[%c0, %c1] : memref<3x3xf64>
    memref.store %c0f, %kernel[%c1, %c0] : memref<3x3xf64>
    memref.store %c0f, %kernel[%c1, %c1] : memref<3x3xf64>
    %0 = "picceler.convolution" (%arg0, %kernel) : (!picceler.image, memref<3x3xf64>) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @ConvolutionImage
// CHECK: call @piccelerCreateImage
// CHECK-COUNT-4: affine.for
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: "picceler.convolution"
// CHECK: return

// -----

func.func @CropImage(%arg0 : !picceler.image) -> !picceler.image {
    %x = "arith.constant"() {value = 30 : i64} : () -> i64
    %y = "arith.constant"() {value = 135 : i64} : () -> i64
    %w = "arith.constant"() {value = 335 : i64} : () -> i64
    %h = "arith.constant"() {value = 240 : i64} : () -> i64
    %0 = "picceler.crop" (%arg0, %x, %y, %w, %h) : (!picceler.image, i64, i64, i64, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @CropImage
// CHECK: call @piccelerCreateImage
// CHECK-COUNT-2: affine.for
// CHECK-NOT: "picceler.crop"
// CHECK: return

