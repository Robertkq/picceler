// RUN: %picceler-opt --picceler-ops-to-func-calls -split-input-file %s | FileCheck %s

func.func @LoadImage() {
    %0 = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    return
}

// CHECK-LABEL: func.func @LoadImage()
// CHECK-NEXT: %[[PATH:.*]] = "picceler.string.const"()
// CHECK-NOT: "picceler.load_image"
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[DATASLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[HSLOT:.*]] = llvm.alloca %[[ONE]] x i64 : (i32) -> !llvm.ptr
// CHECK-NEXT: %[[WSLOT:.*]] = llvm.alloca %[[ONE]] x i64 : (i32) -> !llvm.ptr
// CHECK-NEXT: call @piccelerLoadImage(%[[PATH]], %[[DATASLOT]], %[[HSLOT]], %[[WSLOT]]) : (!picceler.string, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT: %[[DATA:.*]] = llvm.load %[[DATASLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT: %[[H:.*]] = llvm.load %[[HSLOT]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[W:.*]] = llvm.load %[[WSLOT]] : !llvm.ptr -> i64
// CHECK-NEXT: %[[POISON:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[FOUR:.*]] = arith.constant 4 : i64
// CHECK-NEXT: %[[ROWSTRIDE:.*]] = arith.muli %[[W]], %[[FOUR]] : i64
// CHECK-NEXT: %[[S0:.*]] = llvm.insertvalue %[[DATA]], %[[POISON]][0]
// CHECK-NEXT: %[[S1:.*]] = llvm.insertvalue %[[DATA]], %[[S0]][1]
// CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[S2:.*]] = llvm.insertvalue %[[ZERO]], %[[S1]][2]
// CHECK-NEXT: %[[S3:.*]] = llvm.insertvalue %[[H]], %[[S2]][3, 0]
// CHECK-NEXT: %[[S4:.*]] = llvm.insertvalue %[[W]], %[[S3]][3, 1]
// CHECK-NEXT: %[[FOUR2:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[S5:.*]] = llvm.insertvalue %[[FOUR2]], %[[S4]][3, 2]
// CHECK-NEXT: %[[S6:.*]] = llvm.insertvalue %[[ROWSTRIDE]], %[[S5]][4, 0]
// CHECK-NEXT: %[[FOUR3:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT: %[[S7:.*]] = llvm.insertvalue %[[FOUR3]], %[[S6]][4, 1]
// CHECK-NEXT: %[[ONEIDX:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[S8:.*]] = llvm.insertvalue %[[ONEIDX]], %[[S7]][4, 2]
// CHECK-NEXT: %[[LOAD:.*]] = builtin.unrealized_conversion_cast %[[S8]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<?x?x4xi8>
// CHECK-NEXT: return

// -----

func.func @LoadAndShowImage() {
    %0 = "picceler.string.const"() <{ value="cat.png" }> : () -> !picceler.string
    %1 = "picceler.load_image"(%0) : (!picceler.string) -> !picceler.image
    "picceler.show_image"(%1) : (!picceler.image) -> ()
    return
}

// CHECK-LABEL: func.func @LoadAndShowImage()
// CHECK: %[[LOAD:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<?x?x4xi8>
// CHECK-NOT: "picceler.show_image"
// CHECK-NEXT: %[[PTRIDX:.*]] = memref.extract_aligned_pointer_as_index %[[LOAD]] : memref<?x?x4xi8> -> index
// CHECK-NEXT: %[[PTRI64:.*]] = arith.index_cast %[[PTRIDX]] : index to i64
// CHECK-NEXT: %[[PTR:.*]] = llvm.inttoptr %[[PTRI64]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM0:.*]] = memref.dim %[[LOAD]], %[[C0]] : memref<?x?x4xi8>
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[DIM1:.*]] = memref.dim %[[LOAD]], %[[C1]] : memref<?x?x4xi8>
// CHECK-NEXT: %[[H32:.*]] = arith.index_cast %[[DIM0]] : index to i32
// CHECK-NEXT: %[[W32:.*]] = arith.index_cast %[[DIM1]] : index to i32
// CHECK-NEXT: call @piccelerShowImage(%[[PTR]], %[[W32]], %[[H32]]) : (!llvm.ptr, i32, i32) -> ()
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
// CHECK: %[[LOAD:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<?x?x4xi8>
// CHECK-NOT: "picceler.show_image"
// CHECK: call @piccelerShowImage(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i32) -> ()
// CHECK-NEXT: %[[OUTPATH:.*]] = "picceler.string.const"() <{value = "output.png"}>
// CHECK-NOT: "picceler.save_image"
// CHECK-NEXT: %[[PTRIDX2:.*]] = memref.extract_aligned_pointer_as_index %[[LOAD]] : memref<?x?x4xi8> -> index
// CHECK-NEXT: %[[PTRI64_2:.*]] = arith.index_cast %[[PTRIDX2]] : index to i64
// CHECK-NEXT: %[[PTR2:.*]] = llvm.inttoptr %[[PTRI64_2]] : i64 to !llvm.ptr
// CHECK-NEXT: %[[C0_2:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM0_2:.*]] = memref.dim %[[LOAD]], %[[C0_2]] : memref<?x?x4xi8>
// CHECK-NEXT: %[[H32_2:.*]] = arith.index_cast %[[DIM0_2]] : index to i32
// CHECK-NEXT: %[[C1_2:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[DIM1_2:.*]] = memref.dim %[[LOAD]], %[[C1_2]] : memref<?x?x4xi8>
// CHECK-NEXT: %[[W32_2:.*]] = arith.index_cast %[[DIM1_2]] : index to i32
// CHECK-NEXT: call @piccelerSaveImage(%[[PTR2]], %[[W32_2]], %[[H32_2]], %[[OUTPATH]]) : (!llvm.ptr, i32, i32, !picceler.string) -> ()
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
