#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace picceler {

/**
 * @brief A generic pattern that updates the types of operations based on a provided type converter.
 * Used in KernelToMemref and OpsToFuncCalls passes to ensure that operations have the correct types after type
 * conversion.
 */
struct GenericTypeUpdatePattern : public mlir::ConversionPattern {
  GenericTypeUpdatePattern(mlir::TypeConverter &converter, mlir::MLIRContext *context)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override;
};

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
