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

