#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace picceler {

/**
 * @name The following functions create instances of the various passes used in the compilation process.
 * Each function corresponds to a specific pass that transforms the IR in a particular way.
 * @{
 */
std::unique_ptr<mlir::Pass> createPiccelerKernelToMemrefPass();
std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass();
std::unique_ptr<mlir::Pass> createPiccelerToAffinePass();
std::unique_ptr<mlir::Pass> createPiccelerToLLVMIRPass();
std::unique_ptr<mlir::Pass> createPiccelerFiltersToConvPass();

/** @} */

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "piccelerPasses.h.inc"

} // namespace picceler
