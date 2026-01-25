include "mlir/Pass/PassBase.td"

def PiccelerFiltersToConv : Pass<"picceler-filters-to-conv", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Filter Ops to Convolution Ops";
  let constructor = "picceler::createPiccelerFiltersToConvPass()";
}

def PiccelerKernelToMemref : Pass<"picceler-kernel-to-memref", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Kernel Ops to Memref Ops";
  let constructor = "picceler::createPiccelerKernelToMemrefPass()";
}

def PiccelerToAffine : Pass<"picceler-to-affine", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Computational Ops to Affine";
  let constructor = "picceler::createPiccelerToAffinePass()";
}

def PiccelerOpsToFuncCalls : Pass<"picceler-ops-to-func-calls", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Heavy-Ops to function calls";
  let constructor = "picceler::createPiccelerOpsToFuncCallsPass()";
}

def PiccelerToLLVMIR : Pass<"picceler-to-llvmir", "mlir::ModuleOp"> {
  let summary = "Lower Picceler Ops to LLVM IR";
  let constructor = "picceler::createPiccelerToLLVMIRPass()";
}