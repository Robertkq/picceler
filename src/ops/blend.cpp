#include "channels.h"
#include "ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace picceler {

/**
 * @brief Verifies the BlendOp to ensure that the weight is a constant in the range [0.0, 1.0].
 */
mlir::LogicalResult BlendOp::verify() {
  auto weightValue = getWeight();
  auto constWeight = weightValue.getDefiningOp<mlir::arith::ConstantFloatOp>();
  if (!constWeight) {
    return emitOpError("weight must be a compile-time constant");
  }

  double weight = constWeight.value().convertToDouble();
  if (weight < 0.0 || weight > 1.0) {
    return emitOpError("weight must be in the range [0.0, 1.0]");
  }

  return mlir::success();
}

/**
 * @brief Canonicalization pattern for the BlendOp that replaces blend(img, img, weight) with img if both input images
 * are the same.
 */
struct SameOperandPattern : public mlir::OpRewritePattern<BlendOp> {
  using mlir::OpRewritePattern<BlendOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(BlendOp op, mlir::PatternRewriter &rewriter) const override {
    auto input1 = op.getInput1();
    auto input2 = op.getInput2();

    if (input1 == input2) {
      rewriter.replaceOp(op, input1);
      return mlir::success();
    }

    return mlir::failure();
  }
};

/**
 * @brief Registers canonicalization patterns for the BlendOp.
 * Adds the SameOperandPattern to the provided set of rewrite patterns.
 * which performs blend(img, img, weight) -> img if both input images are the same
 *
 * @param results The set of rewrite patterns to which the canonicalization patterns will be added.
 * @param context The MLIR context in which the patterns are registered.
 */
void BlendOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<SameOperandPattern>(context);
}

/**
 * @brief Folds the BlendOp if the weight is a constant equal to 0.0 or 1.0.
 * In these cases, the BlendOp can be replaced with one of its input images, as blending with a weight of 0.0 or 1.0
 * results in one of the input images.
 *
 * @param adaptor The FoldAdaptor providing access to the operands and attributes of the BlendOp.
 * @return mlir::OpFoldResult The folded result, which is one of the input images if the weight is 0.0 or 1.0, or an
 * empty result if no folding is possible.
 */
mlir::OpFoldResult BlendOp::fold(FoldAdaptor adaptor) {
  if (auto weightAttr = llvm::dyn_cast_or_null<mlir::FloatAttr>(adaptor.getWeight())) {
    double weight = weightAttr.getValueAsDouble();
    if (weight == 0.0)
      return getInput1();
    if (weight == 1.0)
      return getInput2();
  }
  return {};
}

mlir::Value BlendOp::transformPixels(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhsPixel,
                                     mlir::Value rhsPixel, Channel ch) {
  auto f64Type = builder.getF64Type();
  auto i8Type = builder.getI8Type();

  mlir::Value lhsPixelAsF64 = builder.create<mlir::arith::UIToFPOp>(loc, f64Type, lhsPixel);
  mlir::Value rhsPixelAsF64 = builder.create<mlir::arith::UIToFPOp>(loc, f64Type, rhsPixel);
  mlir::Value weightValue = getWeight();
  mlir::Value one = createFloatConstant(builder, loc, 1.0);

  mlir::Value lhsPart = builder.create<mlir::arith::MulFOp>(loc, lhsPixelAsF64, weightValue);
  mlir::Value oneMinusWeight = builder.create<mlir::arith::SubFOp>(loc, one, weightValue);
  mlir::Value rhsPart = builder.create<mlir::arith::MulFOp>(loc, rhsPixelAsF64, oneMinusWeight);
  mlir::Value blendedPixel = builder.create<mlir::arith::AddFOp>(loc, lhsPart, rhsPart);

  mlir::Value zero = createFloatConstant(builder, loc, 0.0);
  mlir::Value maxByte = createFloatConstant(builder, loc, 255.0);
  mlir::Value clampedLow = builder.create<mlir::arith::MaximumFOp>(loc, blendedPixel, zero);
  mlir::Value clampedHigh = builder.create<mlir::arith::MinimumFOp>(loc, clampedLow, maxByte);
  return builder.create<mlir::arith::FPToUIOp>(loc, i8Type, clampedHigh);
}

} // namespace picceler