#include "ops.h"
#include "types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace picceler {

mlir::Value DilateOp::initializeAccumulator(mlir::OpBuilder &builder, mlir::Location loc) {
  return createFloatConstant(builder, loc, -std::numeric_limits<double>::infinity());
}

mlir::Value DilateOp::accumulate(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value currentAcc,
                                 mlir::Value pixelValue, mlir::Value optionalKernelValue) {
  auto img = pixelValue;
  (void)img;
  (void)optionalKernelValue;
  return builder.create<mlir::arith::MaximumFOp>(loc, currentAcc, pixelValue).getResult();
}

mlir::Value DilateOp::finalizeAccumulator(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value finalAcc) {
  (void)builder;
  (void)loc;
  return finalAcc;
}

std::pair<uint64_t, uint64_t> DilateOp::getNeighborhoodSize(mlir::ArrayRef<mlir::Value> operands) {
  if (operands.size() < 2) {
    return {0, 0};
  }

  auto img = operands[0];
  auto radiusOperand = operands[1];
  (void)img;

  auto constRadius = radiusOperand.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constRadius) {
    return {0, 0};
  }

  auto neighborhoodSize = static_cast<uint64_t>(2 * constRadius.value() + 1);
  return {neighborhoodSize, neighborhoodSize};
}

} // namespace picceler