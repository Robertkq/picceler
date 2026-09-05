// ===- passes.td - Picceler passes ------------------------------*- tablegen -*-===//
//
// This file defines the Picceler dialect passes.
//
// ===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Picceler Passes
//===----------------------------------------------------------------------===//

def PiccelerFiltersToConv : Pass<"picceler-filters-to-conv", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Filter Ops to Convolution Ops";
  let constructor = "picceler::createPiccelerFiltersToConvPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
    ];
}

def PiccelerKernelToMemref : Pass<"picceler-kernel-to-memref", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Kernel Ops to Memref Ops";
  let constructor = "picceler::createPiccelerKernelToMemrefPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
    ];
}

def PiccelerToAffine : Pass<"picceler-to-affine", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Computational Ops to Affine";
  let constructor = "picceler::createPiccelerToAffinePass()";

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect"
    ];
}

def PiccelerOpsToFuncCalls : Pass<"picceler-ops-to-func-calls", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Heavy-Ops to function calls";
  let constructor = "picceler::createPiccelerOpsToFuncCallsPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::LLVM::LLVMDialect"
    ];
}

def PiccelerToLLVMIR : Pass<"picceler-to-llvmir", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Ops to LLVM IR";
  let constructor = "picceler::createPiccelerToLLVMIRPass()";
}

def PiccelerAddProfiling : Pass<"picceler-add-profiling", "mlir::ModuleOp"> {
  let summary = "Wrap every Picceler op with piccelerTraceBegin/piccelerTraceEnd calls (only added when --profile is passed)";
  let constructor = "picceler::createPiccelerAddProfilingPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
    ];
}