// RUN: %picceler-opt --picceler-to-affine -split-input-file %s | FileCheck %s

func.func @RotateImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %angle = "arith.constant"() {value = 90 : i64} : () -> i64
    %0 = "picceler.rotate" (%arg0, %angle) : (memref<?x?x4xi8>, i64) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @RotateImage
// CHECK: memref.alloc
// CHECK: affine.parallel
// CHECK: arith.select
// CHECK-NOT: "picceler.rotate"
// CHECK: return

// -----

func.func @DiffImages(%arg0 : memref<?x?x4xi8>, %arg1 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %0 = "picceler.diff" (%arg0, %arg1) : (memref<?x?x4xi8>, memref<?x?x4xi8>) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @DiffImages
// CHECK: func.call @abort
// CHECK: memref.alloc
// CHECK: affine.parallel
// CHECK: arith.extui
// CHECK: arith.subi
// CHECK: arith.select
// CHECK-NOT: "picceler.diff"
// CHECK: return

// -----

func.func @BlendImages(%arg0 : memref<?x?x4xi8>, %arg1 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %weight = "arith.constant"() {value = 0.5 : f64} : () -> f64
    %0 = "picceler.blend" (%arg0, %arg1, %weight) : (memref<?x?x4xi8>, memref<?x?x4xi8>, f64) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @BlendImages
// CHECK: memref.alloc
// CHECK: affine.parallel
// CHECK: arith.uitofp
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.fptoui
// CHECK-NOT: "picceler.blend"
// CHECK: return

// -----

func.func @DilateImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %radius = "arith.constant"() {value = 1 : i64} : () -> i64
    %0 = "picceler.dilate" (%arg0, %radius) : (memref<?x?x4xi8>, i64) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @DilateImage
// CHECK: memref.alloc
// CHECK-COUNT-2: affine.parallel
// CHECK: arith.maximumf
// CHECK-NOT: "picceler.dilate"
// CHECK: return

// -----

func.func @ErodeImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %radius = "arith.constant"() {value = 1 : i64} : () -> i64
    %0 = "picceler.erode" (%arg0, %radius) : (memref<?x?x4xi8>, i64) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @ErodeImage
// CHECK: memref.alloc
// CHECK-COUNT-2: affine.parallel
// CHECK: arith.minimumf
// CHECK-NOT: "picceler.erode"
// CHECK: return

// -----

func.func @ConvolutionImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %c0 = "arith.constant"() {value = 0 : index} : () -> index
    %c1 = "arith.constant"() {value = 1 : index} : () -> index
    %c0f = "arith.constant"() {value = 0.0 : f64} : () -> f64
    %kernel = memref.alloca() : memref<3x3xf64>
    memref.store %c0f, %kernel[%c0, %c0] : memref<3x3xf64>
    memref.store %c0f, %kernel[%c0, %c1] : memref<3x3xf64>
    memref.store %c0f, %kernel[%c1, %c0] : memref<3x3xf64>
    memref.store %c0f, %kernel[%c1, %c1] : memref<3x3xf64>
    %0 = "picceler.convolution" (%arg0, %kernel) : (memref<?x?x4xi8>, memref<3x3xf64>) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @ConvolutionImage
// CHECK: memref.alloc
// CHECK-COUNT-2: affine.parallel
// CHECK: memref.load
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: "picceler.convolution"
// CHECK: return

// -----

func.func @CropImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
    %x = "arith.constant"() {value = 30 : i64} : () -> i64
    %y = "arith.constant"() {value = 135 : i64} : () -> i64
    %w = "arith.constant"() {value = 335 : i64} : () -> i64
    %h = "arith.constant"() {value = 240 : i64} : () -> i64
    %0 = "picceler.crop" (%arg0, %x, %y, %w, %h) : (memref<?x?x4xi8>, i64, i64, i64, i64) -> memref<?x?x4xi8>
    return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @CropImage
// CHECK: memref.alloc
// CHECK: affine.parallel
// CHECK-NOT: "picceler.crop"
// CHECK: return

// -----

func.func @InvertImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
  %0 = "picceler.invert"(%arg0) : (memref<?x?x4xi8>) -> memref<?x?x4xi8>
  return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @InvertImage
// CHECK: memref.alloc
// CHECK: affine.parallel
// CHECK: arith.subf
// CHECK-NOT: picceler.invert
// CHECK: return

// -----

func.func @BrightnessImage(%arg0 : memref<?x?x4xi8>) -> memref<?x?x4xi8> {
  %val = "arith.constant"() {value = 50 : i64} : () -> i64
  %0 = "picceler.brightness"(%arg0, %val) : (memref<?x?x4xi8>, i64) -> memref<?x?x4xi8>
  return %0 : memref<?x?x4xi8>
}

// CHECK-LABEL: func.func @BrightnessImage
// CHECK: memref.alloc
// CHECK: affine.parallel
// CHECK: arith.addf
// CHECK-NOT: picceler.brightness
// CHECK: return
