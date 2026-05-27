// RUN: ! %picceler-opt --picceler-to-affine %s 2>&1 | FileCheck %s

func.func @RotateImageInvalidAngle(%arg0 : !picceler.image) -> !picceler.image {
    %angle = "arith.constant"() {value = 45 : i64} : () -> i64
    %0 = "picceler.rotate" (%arg0, %angle) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK: error: 'picceler.rotate' op angle must be a multiple of 90 degrees
