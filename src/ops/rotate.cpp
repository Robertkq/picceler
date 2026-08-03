#include "ops.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <spdlog/spdlog.h>

namespace picceler {

/**
 * @brief Verifies the RotateOp to ensure that the angle is a multiple of 90 degrees if it's a constant integer.
 *
 * @return mlir::success() if the verification passes, otherwise emits an error and returns mlir::failure().
 */
mlir::LogicalResult RotateOp::verify() {
  llvm::APInt angleVal;
  if (mlir::matchPattern(getAngle(), mlir::m_ConstantInt(&angleVal))) {
    int64_t angle = angleVal.getSExtValue();
    if (angle % 90 != 0) {
      return emitOpError("angle must be a multiple of 90, got ") << angle;
    }
  }

  return mlir::success();
}

struct ChainRotatePattern : public mlir::OpRewritePattern<RotateOp> {
  using mlir::OpRewritePattern<RotateOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(RotateOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto prevRotateOp = input.getDefiningOp<RotateOp>();
    if (!prevRotateOp) {
      return mlir::failure();
    }

    llvm::APInt constantAnglePrev;
    llvm::APInt constantAngleCurrent;
    if (mlir::matchPattern(prevRotateOp.getAngle(), mlir::m_ConstantInt(&constantAnglePrev)) &&
        mlir::matchPattern(op.getAngle(), mlir::m_ConstantInt(&constantAngleCurrent))) {
      auto combined = constantAnglePrev.getSExtValue() + constantAngleCurrent.getSExtValue();
      auto normalized = ((combined % 360) + 360) % 360;
      auto newConstantAngle = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), normalized, 64);
      rewriter.replaceOpWithNewOp<RotateOp>(op, op.getType(), prevRotateOp.getInput(), newConstantAngle);
    } else {
      auto newAngle = rewriter.create<mlir::arith::AddIOp>(op.getLoc(), prevRotateOp.getAngle(), op.getAngle());
      rewriter.replaceOpWithNewOp<RotateOp>(op, op.getType(), prevRotateOp.getInput(), newAngle);
    }

    spdlog::debug("ChainRotatePattern applied: rotate(rotate(img, a1), a2) -> rotate(img, a1 + a2)");

    return mlir::success();
  }
};

/**
 * @brief Registers canonicalization patterns for the RotateOp.
 * Adds the ChainRotatePattern to the provided set of rewrite patterns.
 * which performs rotate(rotate(img, a1), a2) -> rotate(img, a1 + a2)
 * either at compile time if both angles are constants, or at runtime if one or both angles are dynamic.
 *
 * @param results The set of rewrite patterns to which the canonicalization patterns will be added.
 * @param context The MLIR context in which the patterns are registered.
 */
void RotateOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<ChainRotatePattern>(context);
}

mlir::OpFoldResult RotateOp::fold(FoldAdaptor adaptor) {
  if (auto angleAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getAngle())) {
    int64_t angle = angleAttr.getInt();
    int64_t normalizedAngle = ((angle % 360) + 360) % 360;
    if (normalizedAngle == 0) {
      return getInput(); // Fold away rotate(img, 0/360/720/-360) -> img
    }
  }
  return {};
}

} // namespace picceler