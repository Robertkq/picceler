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

