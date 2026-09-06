// RUN: %picceler-opt --picceler-add-profiling -split-input-file %s | FileCheck %s

func.func @LoadAndBrighten() {
    %0 = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    %2 = arith.constant 10 : i64
    %3 = "picceler.brightness"(%1, %2) : (!picceler.image, i64) -> !picceler.image
    return
}

// CHECK-DAG: func.func private @piccelerTraceBegin(!picceler.string, i32, i16)
// CHECK-DAG: func.func private @piccelerTraceEnd(!picceler.string, i32, i16)

// CHECK-LABEL: func.func @LoadAndBrighten()
// CHECK-NEXT: %[[PATH:.*]] = "picceler.string.const"() <{value = "cat.png"}>

// CHECK-NEXT: %[[NAME0:.*]] = "picceler.string.const"() <{value = "picceler.load_image"}>
// CHECK-NEXT: %[[IDX0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[TRACK0:.*]] = arith.constant 0 : i16
// CHECK-NEXT: call @piccelerTraceBegin(%[[NAME0]], %[[IDX0]], %[[TRACK0]]) : (!picceler.string, i32, i16) -> ()
// CHECK-NEXT: %[[IMG:.*]] = "picceler.load_image"(%[[PATH]])
// CHECK-NEXT: call @piccelerTraceEnd(%[[NAME0]], %[[IDX0]], %[[TRACK0]]) : (!picceler.string, i32, i16) -> ()

// CHECK-NEXT: %[[VAL:.*]] = arith.constant 10 : i64
// CHECK-NEXT: %[[NAME1:.*]] = "picceler.string.const"() <{value = "picceler.brightness"}>
// CHECK-NEXT: %[[IDX1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[TRACK1:.*]] = arith.constant 0 : i16
// CHECK-NEXT: call @piccelerTraceBegin(%[[NAME1]], %[[IDX1]], %[[TRACK1]]) : (!picceler.string, i32, i16) -> ()
// CHECK-NEXT: %{{.*}} = "picceler.brightness"(%[[IMG]], %[[VAL]])
// CHECK-NEXT: call @piccelerTraceEnd(%[[NAME1]], %[[IDX1]], %[[TRACK1]]) : (!picceler.string, i32, i16) -> ()
// CHECK-NEXT: return

// -----

func.func @GaussianBlur(%arg0 : !picceler.image) -> !picceler.image {
    %radius = arith.constant 3 : i64
    %0 = "picceler.gaussian_blur"(%arg0, %radius) : (!picceler.image, i64) -> !picceler.image
    return %0 : !picceler.image
}

// CHECK-LABEL: func.func @GaussianBlur(
// CHECK: %[[NAME:.*]] = "picceler.string.const"() <{value = "picceler.gaussian_blur"}>
// CHECK-NEXT: %[[IDX:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[TRACK:.*]] = arith.constant 0 : i16
// CHECK-NEXT: call @piccelerTraceBegin(%[[NAME]], %[[IDX]], %[[TRACK]])
// CHECK-NEXT: %[[RESULT:.*]] = "picceler.gaussian_blur"
// CHECK-NEXT: call @piccelerTraceEnd(%[[NAME]], %[[IDX]], %[[TRACK]])
// CHECK-NEXT: return %[[RESULT]]

// -----

// Blocking/non-compute ops (show_image, read_number, read_string, print) and kernel.const stay
// unwrapped; load_image/save_image and the compute op in between still get traced.

func.func @ExcludedOps(%arg0 : !picceler.image) {
    %path = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %img = "picceler.load_image"(%path) : (!picceler.string) -> !picceler.image
    %kernel = "picceler.kernel.const"() <{ values = dense<1.0> : tensor<3x3xf64> }> : () -> !picceler.kernel<3 x 3>
    %blurred = "picceler.convolution"(%img, %kernel) : (!picceler.image, !picceler.kernel<3 x 3>) -> !picceler.image
    "picceler.show_image"(%blurred) : (!picceler.image) -> ()
    %prompt = "picceler.string.const"() <{ value="save?" }> : () -> !picceler.string
    %answer = "picceler.read_string"(%prompt) : (!picceler.string) -> !picceler.string
    %n = "picceler.read_number"(%prompt) : (!picceler.string) -> f64
    %fmt = "picceler.string.const"() <{ value="{}\0A" }> : () -> !picceler.string
    "picceler.print"(%fmt, %answer) : (!picceler.string, !picceler.string) -> ()
    "picceler.save_image"(%blurred, %answer) : (!picceler.image, !picceler.string) -> ()
    return
}

// A wrapped op gets a "picceler.string.const" holding its exact name; an excluded op doesn't, so
// checking for that name const (rather than the call site, which only ever holds an SSA value) is
// what actually distinguishes wrapped from unwrapped here.

// CHECK-LABEL: func.func @ExcludedOps(
// CHECK: "picceler.string.const"() <{value = "picceler.load_image"}>
// CHECK-NOT: "picceler.string.const"() <{value = "picceler.kernel.const"}>
// CHECK: "picceler.string.const"() <{value = "picceler.convolution"}>
// CHECK-NOT: "picceler.string.const"() <{value = "picceler.show_image"}>
// CHECK-NOT: "picceler.string.const"() <{value = "picceler.read_string"}>
// CHECK-NOT: "picceler.string.const"() <{value = "picceler.read_number"}>
// CHECK-NOT: "picceler.string.const"() <{value = "picceler.print"}>
// CHECK: "picceler.string.const"() <{value = "picceler.save_image"}>
// CHECK: "picceler.save_image"
