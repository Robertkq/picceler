#include "ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <limits>

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

Result<std::pair<mlir::Value, mlir::Value>> DilateOp::getNeighborhoodSize(mlir::OpBuilder &builder, mlir::Location loc,
                                                                          mlir::ArrayRef<mlir::Value> operands) {
  if (operands.size() < 2) {
    return std::unexpected(CompileError("DilateOp requires at least 2 operands: input image and radius"));
  }

  [[maybe_unused]] auto img = operands[0];
  auto radius = operands[1];

  auto doubleRadius = builder.create<mlir::arith::AddIOp>(loc, radius, radius);
  auto neighborhoodSize = builder.create<mlir::arith::AddIOp>(loc, doubleRadius, createIntConstant(builder, loc, 1));
  return std::make_pair(neighborhoodSize, neighborhoodSize);
}

} // namespace picceler