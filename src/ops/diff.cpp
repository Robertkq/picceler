#include "ops.h"
#include "channels.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace picceler {

/**
 * @brief Verifies the DiffOp to ensure that the input images have the same type and that the result image type matches
 * the input image types.
 *
 * @return mlir::success() if the verification passes, otherwise emits an error and returns mlir::failure().
 */
mlir::LogicalResult DiffOp::verify() {
  auto input1 = getInput1();
  auto input2 = getInput2();

  if (input1.getType() != input2.getType()) {
    return emitOpError("Input images must have the same type, got ") << input1.getType() << " and " << input2.getType();
  }

  if (input1.getType() != getResult().getType()) {
    return emitOpError("Result image type must match input image types, got ")
           << getResult().getType() << " and " << input1.getType();
  }

  return mlir::success();
}

/**
 * @brief Registers canonicalization patterns for the DiffOp
 *
 * @param results The set of rewrite patterns to which the canonicalization patterns will be added.
 * @param context The MLIR context in which the patterns are registered.
 */
void DiffOp::getCanonicalizationPatterns(mlir::RewritePatternSet &, mlir::MLIRContext *) {}

mlir::OpFoldResult DiffOp::fold(FoldAdaptor) { return {}; }

mlir::Value DiffOp::transformPixels(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhsPixel,
                                    mlir::Value rhsPixel, Channel ch) {
  if (static_cast<int>(ch) == 3) {
    // For the alpha channel, we will not apply blending and just take the value from the first image.
    return lhsPixel;
  }
  auto extLhsByte = builder.create<mlir::arith::ExtUIOp>(loc, builder.getI16Type(), lhsPixel);
  auto extRhsByte = builder.create<mlir::arith::ExtUIOp>(loc, builder.getI16Type(), rhsPixel);

  auto diff = builder.create<mlir::arith::SubIOp>(loc, extLhsByte, extRhsByte);
  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 16);
  auto isNegative = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, diff, zero);

  auto negDiff = builder.create<mlir::arith::SubIOp>(loc, zero, diff);
  auto absDiff = builder.create<mlir::arith::SelectOp>(loc, isNegative, negDiff, diff);

  auto truncatedDiff = builder.create<mlir::arith::TruncIOp>(loc, builder.getI8Type(), absDiff);

  return truncatedDiff;
}

} // namespace picceler