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