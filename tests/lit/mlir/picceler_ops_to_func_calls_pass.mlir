// RUN: %picceler-opt --picceler-ops-to-func-calls -split-input-file %s | FileCheck %s

func.func @LoadImage() {
    %0 = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    return
}

// CHECK-LABEL: func.func @LoadImage()
// CHECK-NEXT: %[[PATH:.*]] = "picceler.string.const"() 
// CHECK-NOT: "picceler.load_image"
// CHECK-NEXT: %[[LOAD:.*]] = call @piccelerLoadImage(%[[PATH]]) : (!picceler.string) -> !picceler.image
// CHECK-NEXT: return

// -----

func.func @LoadAndShowImage() {
    %0 = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    "picceler.show_image"(%1) : (!picceler.image) -> ()
    return
}

// CHECK-LABEL: func.func @LoadAndShowImage()
// CHECK-NEXT: %[[PATH:.*]] = "picceler.string.const"() 
// CHECK-NOT: "picceler.load_image"
// CHECK-NEXT: %[[LOAD:.*]] = call @piccelerLoadImage(%[[PATH]]) : (!picceler.string) -> !picceler.image
// CHECK-NOT: "picceler.show_image"
// CHECK-NEXT: call @piccelerShowImage(%[[LOAD]]) : (!picceler.image) -> ()
// CHECK-NEXT: return

// -----

func.func @LoadShowSaveImage() {
    %0 = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    "picceler.show_image"(%1) : (!picceler.image) -> ()
    %2 = "picceler.string.const"() <{ value="output.png" }> : () -> !picceler.string
    "picceler.save_image"(%1, %2) : (!picceler.image, !picceler.string) -> ()
    return
}

// CHECK-LABEL: func.func @LoadShowSaveImage()
// CHECK-NEXT: %[[PATH:.*]] = "picceler.string.const"() 
// CHECK-NOT: "picceler.load_image"
// CHECK-NEXT: %[[LOAD:.*]] = call @piccelerLoadImage(%[[PATH]]) : (!picceler.string) -> !picceler.image
// CHECK-NOT: "picceler.show_image"
// CHECK-NEXT: call @piccelerShowImage(%[[LOAD]]) : (!picceler.image) -> ()
// CHECK-NEXT: %[[OUTPATH:.*]] = "picceler.string.const"() 
// CHECK-NOT: "picceler.save_image"
// CHECK-NEXT: call @piccelerSaveImage(%[[LOAD]], %[[OUTPATH]]) : (!picceler.image, !picceler.string) -> ()
// CHECK-NEXT: return

// -----

func.func @ReadStringAndNumber() {
    %0 = "picceler.string.const"() <{ value="Enter a string: " }> : () -> !picceler.string
    %1 = "picceler.read_string"(%0) : (!picceler.string) -> !picceler.string
    %2 = "picceler.string.const"() <{ value="Enter a number: " }> : () -> !picceler.string
    %3 = "picceler.read_number"(%2) : (!picceler.string) -> f64
    return
}

// CHECK-LABEL: func.func @ReadStringAndNumber()
// CHECK-NEXT: %[[PROMPT1:.*]] = "picceler.string.const"() 
// CHECK-NEXT: %[[READ_STR:.*]] = call @piccelerReadString(%[[PROMPT1]]) : (!picceler.string) -> !picceler.string
// CHECK-NEXT: %[[PROMPT2:.*]] = "picceler.string.const"() 
// CHECK-NEXT: %[[READ_NUM:.*]] = call @piccelerReadNumber(%[[PROMPT2]]) : (!picceler.string) -> f64
// CHECK-NEXT: return

// -----

func.func @PrintSimpleString() {
    %fmt = "picceler.string.const"() <{ value = "Hello, World!" }> : () -> !picceler.string
    "picceler.print"(%fmt) : (!picceler.string) -> ()
    return
}

// CHECK-LABEL: func.func @PrintSimpleString()
// CHECK-DAG: %[[STR:.*]] = "picceler.string.const"() <{value = "Hello, World!"}>
// CHECK:     call @piccelerPrintString(%[[STR]])
// CHECK:     return

// -----

func.func @Print2Parts1ArgString() {
    %fmt1 = "picceler.string.const"() <{ value = "Hello, {}!" }> : () -> !picceler.string
    %fmt2 = "picceler.string.const"() <{ value = "World" }> : () -> !picceler.string
    "picceler.print"(%fmt1, %fmt2) : (!picceler.string, !picceler.string) -> ()
    return
}

// CHECK-LABEL: func.func @Print2Parts1ArgString()
// CHECK-DAG: %[[PART2:.*]] = "picceler.string.const"() <{value = "World"}>
// CHECK-DAG: %[[PART1:.*]] = "picceler.string.const"() <{value = "Hello, "}>
// CHECK:      call @piccelerPrintString(%[[PART1]])
// CHECK-NEXT: call @piccelerPrintString(%[[PART2]])
// CHECK-NEXT: %[[PART3:.*]] = "picceler.string.const"() <{value = "!"}>
// CHECK-NEXT: call @piccelerPrintString(%[[PART3]])
// CHECK-NEXT: return

// -----

func.func @Print2Parts1FloatNewlineTerminated() {
    %fp64 = arith.constant 3.14159 : f64
    %fmt1 = "picceler.string.const"() <{ value = "The value of pi is approximately: {}\n" }> : () -> !picceler.string
    "picceler.print"(%fmt1, %fp64) : (!picceler.string, f64) -> ()
    return
}

// CHECK-LABEL: func.func @Print2Parts1FloatNewlineTerminated()
// CHECK-DAG: %[[FP:.*]] = arith.constant 3.141590e+00 : f64
// CHECK-DAG: %[[PART1:.*]] = "picceler.string.const"() <{value = "The value of pi is approximately: "}>
// CHECK:      call @piccelerPrintString(%[[PART1]])
// CHECK-NEXT: call @piccelerPrintFloat64(%[[FP]])
// CHECK-NEXT: %[[NL:.*]] = "picceler.string.const"() <{value = "\0A"}>
// CHECK-NEXT: call @piccelerPrintString(%[[NL]])
// CHECK-NEXT: return

// -----

func.func @PrintAdjacentPlaceholders() {
    %arg1 = "picceler.string.const"() <{ value = "Foo" }> : () -> !picceler.string
    %arg2 = "picceler.string.const"() <{ value = "Bar" }> : () -> !picceler.string
    %fmt = "picceler.string.const"() <{ value = "{}{}\n" }> : () -> !picceler.string
    "picceler.print"(%fmt, %arg1, %arg2) : (!picceler.string, !picceler.string, !picceler.string) -> ()
    return
}

// CHECK-LABEL: func.func @PrintAdjacentPlaceholders()
// CHECK-DAG: %[[ARG1:.*]] = "picceler.string.const"() <{value = "Foo"}>
// CHECK-DAG: %[[ARG2:.*]] = "picceler.string.const"() <{value = "Bar"}>
// CHECK:      call @piccelerPrintString(%[[ARG1]])
// CHECK-NEXT: call @piccelerPrintString(%[[ARG2]])
// CHECK:      call @piccelerPrintString
// CHECK:      return

// -----

func.func @PrintAdjacentNoNewline() {
    %arg1 = "picceler.string.const"() <{ value = "A" }> : () -> !picceler.string
    %arg2 = "picceler.string.const"() <{ value = "B" }> : () -> !picceler.string
    %fmt = "picceler.string.const"() <{ value = "{}{}" }> : () -> !picceler.string
    "picceler.print"(%fmt, %arg1, %arg2) : (!picceler.string, !picceler.string, !picceler.string) -> ()
    return
}

// CHECK-LABEL: func.func @PrintAdjacentNoNewline()
// CHECK-DAG: %[[ARG1:.*]] = "picceler.string.const"() <{value = "A"}>
// CHECK-DAG: %[[ARG2:.*]] = "picceler.string.const"() <{value = "B"}>
// CHECK:      call @piccelerPrintString(%[[ARG1]])
// CHECK-NEXT: call @piccelerPrintString(%[[ARG2]])
// CHECK-NEXT: return

// -----

func.func @PrintLeadingPlaceholder(%arg0: f64) {
    %fmt = "picceler.string.const"() <{ value = "{} is the output\n" }> : () -> !picceler.string
    "picceler.print"(%fmt, %arg0) : (!picceler.string, f64) -> ()
    return
}

// CHECK-LABEL: func.func @PrintLeadingPlaceholder(
// CHECK-SAME:                                     %[[ARG0:.*]]: f64)
// CHECK:      call @piccelerPrintFloat64(%[[ARG0]])
// CHECK-NEXT: %[[SUFFIX:.*]] = "picceler.string.const"() <{value = " is the output\0A"}>
// CHECK-NEXT: call @piccelerPrintString(%[[SUFFIX]])
// CHECK-NEXT: return
