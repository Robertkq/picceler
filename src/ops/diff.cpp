#include "ops.h"
#include "channels.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace picceler {

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