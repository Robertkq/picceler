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