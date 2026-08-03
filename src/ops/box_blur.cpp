#include "ops.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <spdlog/spdlog.h>

namespace picceler {

/**
 * @brief Verifies the BoxBlurOp to ensure that the input and result types match.
 * @return mlir::success() if the verification passes, otherwise emits an error and returns mlir::failure().
 */
mlir::LogicalResult BoxBlurOp::verify() {

  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result types must match");
  }

  llvm::APInt radiusVal;
  if (mlir::matchPattern(getRadius(), mlir::m_ConstantInt(&radiusVal))) {
    int64_t radius = radiusVal.getSExtValue();
    if (radius < 0) {
      return emitOpError("radius must be non-negative, but got ") << radius;
    }
  }

  return mlir::success();
}

/**
 * @brief Registers canonicalization patterns for the BoxBlurOp.
 * No canonicalization patterns currently
 *
 */
void BoxBlurOp::getCanonicalizationPatterns(mlir::RewritePatternSet &, mlir::MLIRContext *) {}

/**
 * @brief Folds away box_blur(img, 0) -> img
 * @param adaptor The adaptor for the box_blur operation.
 * @return The folded result if applicable, otherwise an empty OpFoldResult.
 */
mlir::OpFoldResult BoxBlurOp::fold(FoldAdaptor adaptor) {
  if (auto radiusAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getRadius())) {
    if (radiusAttr.getInt() == 0) {
      return getInput(); // Fold away box_blur(img, 0) -> img
    }
  }
  return {};
}

} // namespace picceler