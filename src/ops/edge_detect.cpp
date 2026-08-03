#include "ops.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <spdlog/spdlog.h>

namespace picceler {

/**
 * @brief Verifies the EdgeDetectOp to ensure that the input and result types match.
 *
 * @return mlir::success() if the verification passes, otherwise emits an error and returns mlir::failure().
 */
mlir::LogicalResult EdgeDetectOp::verify() {
  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result types must match");
  }

  return mlir::success();
}

struct RemoveInvertPattern : public mlir::OpRewritePattern<EdgeDetectOp> {
  using mlir::OpRewritePattern<EdgeDetectOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(EdgeDetectOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto invertOp = input.getDefiningOp<InvertOp>();
    if (!invertOp) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<EdgeDetectOp>(op, op.getType(), invertOp.getInput());

    spdlog::debug("RemoveInvertPattern applied: edge_detect(invert(img)) -> edge_detect(img)");

    return mlir::success();
  }
};

/**
 * @brief Registers canonicalization patterns for the EdgeDetectOp.
 * Adds a pattern to remove an invert operation that is immediately followed by an edge_detect operation.
 * edge_detect(invert(img)) -> edge_detect(img)
 *
 * @param results The set of rewrite patterns to populate.
 * @param context The MLIR context in which the patterns are registered.
 */
void EdgeDetectOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<RemoveInvertPattern>(context);
}

/**
 * @brief Folder for the EdgeDetectOp.
 * Currently, there are no folding rules implemented for this operation.
 *
 * @return returnsan empty OpFoldResult.
 */
mlir::OpFoldResult EdgeDetectOp::fold(FoldAdaptor) { return {}; }

} // namespace picceler