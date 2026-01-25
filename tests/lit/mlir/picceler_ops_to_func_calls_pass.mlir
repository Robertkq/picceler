// RUN: %picceler-opt --picceler-ops-to-func-calls -split-input-file %s | FileCheck %s

func.func @LoadImage() {
    %0 = picceler.string.const "cat.png" : !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    return
}

// Put all checks together at the bottom of the block
// CHECK-LABEL: func.func @LoadImage()
// CHECK-NEXT:    %[[PATH:.*]] = picceler.string.const "cat.png" : !picceler.string
// CHECK-NOT:     "picceler.load_image"
// CHECK-NEXT:    %[[LOAD:.*]] = call @piccelerLoadImage(%[[PATH]]) : (!picceler.string) -> !picceler.image
// CHECK-NEXT:    return

// -----

func.func @LoadAndShowImage() {
    %0 = picceler.string.const "cat.png" : !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    "picceler.show_image"(%1) : (!picceler.image) -> ()
    return
}

// CHECK-LABEL: func.func @LoadAndShowImage()
// CHECK-NEXT:    %[[PATH:.*]] = picceler.string.const "cat.png" : !picceler.string
// CHECK-NOT:     "picceler.load_image"
// CHECK-NEXT:    %[[LOAD:.*]] = call @piccelerLoadImage(%[[PATH]]) : (!picceler.string) -> !picceler.image
// CHECK-NOT:     "picceler.show_image"
// CHECK-NEXT:    call @piccelerShowImage(%[[LOAD]]) : (!picceler.image) -> ()
// CHECK-NEXT:    return