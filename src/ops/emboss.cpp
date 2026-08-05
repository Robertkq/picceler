#include "ops.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <spdlog/spdlog.h>

namespace picceler {

/**
 * @brief Verifies the EmbossOp to ensure that the input and result types match.
 *
 * @return mlir::success() if the verification passes, otherwise emits an error and returns mlir::failure().
 */
mlir::LogicalResult EmbossOp::verify() {
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result types must match");
  }

  return mlir::success();
}

/**
 * @brief Registers canonicalization patterns for the EmbossOp.
 * No canonicalization patterns currently
 *
 */
void EmbossOp::getCanonicalizationPatterns(mlir::RewritePatternSet &, mlir::MLIRContext *) {}

/**
 * @brief Folder for the EmbossOp.
 * Currently, there are no folding rules implemented for this operation.
 *
 * @return returns an empty OpFoldResult.
 */
mlir::OpFoldResult EmbossOp::fold(FoldAdaptor) { return {}; }

} // namespace picceler