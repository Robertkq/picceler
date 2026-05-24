#include "ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <limits>

namespace picceler {

mlir::Value ErodeOp::initializeAccumulator(mlir::OpBuilder &builder, mlir::Location loc) {
  return createFloatConstant(builder, loc, std::numeric_limits<double>::infinity());
}

mlir::Value ErodeOp::accumulate(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value currentAcc,
                                mlir::Value pixelValue, mlir::Value optionalKernelValue) {
  auto img = pixelValue;
  (void)img;
  (void)optionalKernelValue;
  return builder.create<mlir::arith::MinimumFOp>(loc, currentAcc, pixelValue).getResult();
}

mlir::Value ErodeOp::finalizeAccumulator(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value finalAcc) {
  (void)builder;
  (void)loc;
  return finalAcc;
}

std::pair<uint64_t, uint64_t> ErodeOp::getNeighborhoodSize(mlir::ArrayRef<mlir::Value> operands) {
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