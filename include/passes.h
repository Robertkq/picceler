#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace picceler {

std::unique_ptr<mlir::Pass> createPiccelerKernelToMemrefPass();
std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass();
std::unique_ptr<mlir::Pass> createPiccelerToAffinePass();
std::unique_ptr<mlir::Pass> createPiccelerToLLVMIRPass();
std::unique_ptr<mlir::Pass> createPiccelerFiltersToConvPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "piccelerPasses.h.inc"

} // namespace picceler
