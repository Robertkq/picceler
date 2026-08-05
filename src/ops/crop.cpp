#include "ops.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <spdlog/spdlog.h>

namespace picceler {

/**
 * @brief Verifies the CropOp to ensure that the x and y coordinates are non-negative and that the width and height are
 * strictly positive if they are constant integers.
 */
mlir::LogicalResult CropOp::verify() {
  llvm::APInt xVal;
  llvm::APInt yVal;
  llvm::APInt wVal;
  llvm::APInt hVal;

  if (mlir::matchPattern(getX(), mlir::m_ConstantInt(&xVal)) && xVal.getSExtValue() < 0)
    return emitOpError("x coordinate must be non-negative");

  if (mlir::matchPattern(getY(), mlir::m_ConstantInt(&yVal)) && yVal.getSExtValue() < 0)
    return emitOpError("y coordinate must be non-negative");

  if (mlir::matchPattern(getWidth(), mlir::m_ConstantInt(&wVal)) && wVal.getSExtValue() <= 0)
    return emitOpError("crop width must be strictly positive");

  if (mlir::matchPattern(getHeight(), mlir::m_ConstantInt(&hVal)) && hVal.getSExtValue() <= 0)
    return emitOpError("crop height must be strictly positive");

  return mlir::success();
}

/**
 * @brief Canonicalization pattern for the CropOp that combines nested crop operations into a single crop operation.
 * only applicable when both crop operations have constant integer coordinates and dimensions.
 */
struct CombineNestedCrops : public mlir::OpRewritePattern<CropOp> {
  using OpRewritePattern<CropOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(CropOp outerCrop, mlir::PatternRewriter &rewriter) const override {
    auto innerCrop = outerCrop.getInput().getDefiningOp<CropOp>();
    if (!innerCrop)
      return mlir::failure();

    llvm::APInt x1, y1, w1, h1;
    llvm::APInt x2, y2, w2, h2;

    if (!mlir::matchPattern(innerCrop.getX(), mlir::m_ConstantInt(&x1)) ||
        !mlir::matchPattern(innerCrop.getY(), mlir::m_ConstantInt(&y1)) ||
        !mlir::matchPattern(innerCrop.getWidth(), mlir::m_ConstantInt(&w1)) ||
        !mlir::matchPattern(innerCrop.getHeight(), mlir::m_ConstantInt(&h1)))
      return mlir::failure();

    if (!mlir::matchPattern(outerCrop.getX(), mlir::m_ConstantInt(&x2)) ||
        !mlir::matchPattern(outerCrop.getY(), mlir::m_ConstantInt(&y2)) ||
        !mlir::matchPattern(outerCrop.getWidth(), mlir::m_ConstantInt(&w2)) ||
        !mlir::matchPattern(outerCrop.getHeight(), mlir::m_ConstantInt(&h2)))
      return mlir::failure();

    int64_t x1Val = x1.getSExtValue();
    int64_t y1Val = y1.getSExtValue();
    int64_t w1Val = w1.getSExtValue();
    int64_t h1Val = h1.getSExtValue();

    int64_t x2Val = x2.getSExtValue();
    int64_t y2Val = y2.getSExtValue();
    int64_t w2Val = w2.getSExtValue();
    int64_t h2Val = h2.getSExtValue();

    int64_t fusedX = x1Val + x2Val;
    int64_t fusedY = y1Val + y2Val;
    int64_t fusedW = std::min(w2Val, std::max<int64_t>(0, w1Val - x2Val));
    int64_t fusedH = std::min(h2Val, std::max<int64_t>(0, h1Val - y2Val));

    mlir::Location loc = outerCrop.getLoc();
    auto newX = rewriter.create<mlir::arith::ConstantIntOp>(loc, fusedX, 64);
    auto newY = rewriter.create<mlir::arith::ConstantIntOp>(loc, fusedY, 64);
    auto newW = rewriter.create<mlir::arith::ConstantIntOp>(loc, fusedW, 64);
    auto newH = rewriter.create<mlir::arith::ConstantIntOp>(loc, fusedH, 64);

    rewriter.replaceOpWithNewOp<CropOp>(outerCrop, outerCrop.getType(), innerCrop.getInput(), newX, newY, newW, newH);
    return mlir::success();
  }
};

/**
 * @brief Registers canonicalization patterns for the CropOp.
 * Adds the CombineNestedCrops pattern to the provided set of rewrite patterns.
 * which combines nested crop operations into a single crop operation.
 *
 * @param results The set of rewrite patterns to which the canonicalization patterns will be added.
 * @param context The MLIR context in which the patterns are registered.
 */
void CropOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<CombineNestedCrops>(context);
}

mlir::OpFoldResult CropOp::fold(FoldAdaptor) { return {}; }

} // namespace picceler