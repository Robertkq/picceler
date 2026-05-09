// RUN: %picceler-opt --picceler-kernel-to-memref -split-input-file %s | FileCheck %s


func.func @SmallKernelOfOnes() {
    %0 = "picceler.kernel.const"() <{values = dense<1.000000e+00> : tensor<3x3xf64>}> : () -> !picceler.kernel<3 x 3>
    return 
}

// CHECK-LABEL: func.func @SmallKernelOfOnes()
// CHECK-DAG: %[[KERNEL_VALUE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG: %[[INDEX_0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[INDEX_1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[INDEX_2:.*]] = arith.constant 2 : index
// CHECK: %[[MEMREF:.*]] = memref.alloca() : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_0]], %[[INDEX_0]]]  : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_0]], %[[INDEX_1]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_0]], %[[INDEX_2]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_1]], %[[INDEX_0]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_1]], %[[INDEX_1]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_1]], %[[INDEX_2]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_2]], %[[INDEX_0]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_2]], %[[INDEX_1]]] : memref<3x3xf64>
// CHECK: memref.store %[[KERNEL_VALUE]], %[[MEMREF]][%[[INDEX_2]], %[[INDEX_2]]] : memref<3x3xf64>
// CHECK-NEXT: return

// -----

func.func @VerticalLineKernel() {
    %0 = "picceler.kernel.const"() <{values = dense<[[-1.000000e+00, 2.000000e+00, -1.000000e+00], [-1.000000e+00, 2.000000e+00, -1.000000e+00], [-1.000000e+00, 2.000000e+00, -1.000000e+00]]> : tensor<3x3xf64>}> : () -> !picceler.kernel<3 x 3>
    return 
}

// CHECK-LABEL: func.func @VerticalLineKernel()
// CHECK-DAG: %[[NEG1:.*]] = arith.constant -1.000000e+00 : f64
// CHECK-DAG: %[[POS2:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG: %[[INDEX_0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[INDEX_1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[INDEX_2:.*]] = arith.constant 2 : index
// CHECK: %[[MEMREF:.*]] = memref.alloca() : memref<3x3xf64>
// CHECK: memref.store %[[NEG1]], %[[MEMREF]][%[[INDEX_0]], %[[INDEX_0]]] : memref<3x3xf64>
// CHECK: memref.store %[[POS2]], %[[MEMREF]][%[[INDEX_0]], %[[INDEX_1]]] : memref<3x3xf64>
// CHECK: memref.store %[[NEG1]], %[[MEMREF]][%[[INDEX_0]], %[[INDEX_2]]] : memref<3x3xf64>
// CHECK: memref.store %[[NEG1]], %[[MEMREF]][%[[INDEX_1]], %[[INDEX_0]]] : memref<3x3xf64>
// CHECK: memref.store %[[POS2]], %[[MEMREF]][%[[INDEX_1]], %[[INDEX_1]]] : memref<3x3xf64>
// CHECK: memref.store %[[NEG1]], %[[MEMREF]][%[[INDEX_1]], %[[INDEX_2]]] : memref<3x3xf64>
// CHECK: memref.store %[[NEG1]], %[[MEMREF]][%[[INDEX_2]], %[[INDEX_0]]] : memref<3x3xf64>
// CHECK: memref.store %[[POS2]], %[[MEMREF]][%[[INDEX_2]], %[[INDEX_1]]] : memref<3x3xf64>
// CHECK: memref.store %[[NEG1]], %[[MEMREF]][%[[INDEX_2]], %[[INDEX_2]]] : memref<3x3xf64>
// CHECK-NEXT: return

