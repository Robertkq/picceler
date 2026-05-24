#include "ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace picceler {

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

mlir::Value BlendOp::transformPixels(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhsPixel,
                                     mlir::Value rhsPixel, int offset) {
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