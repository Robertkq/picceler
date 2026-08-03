// RUN: %picceler-opt --picceler-kernel-to-memref -split-input-file %s | FileCheck %s

func.func @SmallKernelOfOnes() -> !picceler.kernel<3 x 3> {
    %0 = "picceler.kernel.const"() <{values = dense<1.000000e+00> : tensor<3x3xf64>}> : () -> !picceler.kernel<3 x 3>
    return %0 : !picceler.kernel<3 x 3>
}

// CHECK-LABEL: func.func @SmallKernelOfOnes() -> memref<3x3xf64> {
// CHECK:         %[[MEMREF:.*]] = memref.alloca() : memref<3x3xf64>
// CHECK-DAG:     %[[V0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:     %[[R0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.store %[[V0]], %[[MEMREF]][%[[R0]], %[[C0]]] : memref<3x3xf64>
// CHECK-DAG:     %[[R0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[V0_1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:         memref.store %[[V0_1]], %[[MEMREF]][%[[R0_1]], %[[C1]]] : memref<3x3xf64>
// CHECK:         return %[[MEMREF]] : memref<3x3xf64>
// CHECK:       }

// -----

func.func @VerticalLineKernel() -> !picceler.kernel<3 x 3> {
    %0 = "picceler.kernel.const"() <{values = dense<[[-1.000000e+00, 2.000000e+00, -1.000000e+00], [-1.000000e+00, 2.000000e+00, -1.000000e+00], [-1.000000e+00, 2.000000e+00, -1.000000e+00]]> : tensor<3x3xf64>}> : () -> !picceler.kernel<3 x 3>
    return %0 : !picceler.kernel<3 x 3>
}

// CHECK-LABEL: func.func @VerticalLineKernel() -> memref<3x3xf64> {
// CHECK:         %[[MEMREF:.*]] = memref.alloca() : memref<3x3xf64>
// CHECK-DAG:     %[[NEG1:.*]] = arith.constant -1.000000e+00 : f64
// CHECK-DAG:     %[[R0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         memref.store %[[NEG1]], %[[MEMREF]][%[[R0]], %[[C0]]] : memref<3x3xf64>
// CHECK-DAG:     %[[POS2:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:     %[[R0_1:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         memref.store %[[POS2]], %[[MEMREF]][%[[R0_1]], %[[C1]]] : memref<3x3xf64>
// CHECK:         return %[[MEMREF]] : memref<3x3xf64>
// CHECK:       }



