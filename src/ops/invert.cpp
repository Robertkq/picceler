#include "ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <spdlog/spdlog.h>

namespace picceler {

mlir::LogicalResult InvertOp::verify() {

  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result types must match");
  }

  return mlir::success();
}

struct ChainedInvertPattern : public mlir::OpRewritePattern<InvertOp> {
  using OpRewritePattern<InvertOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(InvertOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();

    if (auto definingOp = input.getDefiningOp<InvertOp>()) {
      auto newInput = definingOp.getInput();
      rewriter.replaceOp(op, newInput);
      return mlir::success();
    }

    return mlir::success();
  }
};

void InvertOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<ChainedInvertPattern>(context);
}

mlir::OpFoldResult InvertOp::fold(FoldAdaptor adaptor) { return mlir::OpFoldResult(); }

} // namespace picceler