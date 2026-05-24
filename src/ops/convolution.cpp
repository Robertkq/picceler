#include "ops.h"
#include "types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace picceler {

llvm::LogicalResult KernelConstOp::verify() {
  auto valuesAttr = getValues();
  auto kernel = getResult().getType();

  auto expectedSize = kernel.getRows() * kernel.getCols();
  auto actualSize = valuesAttr.size();

  if (expectedSize != actualSize) {
    return emitOpError("number of values (") << actualSize << ") does not match kernel (" << kernel.getRows() << "x"
                                             << kernel.getCols() << " = " << expectedSize << ")";
  }

  return mlir::success();
}

std::pair<uint64_t, uint64_t> getKernelNeighborhoodSize(mlir::Value kernelOperand) {
  if (auto kernelMemRefType = mlir::dyn_cast<mlir::MemRefType>(kernelOperand.getType())) {
    if (kernelMemRefType.getRank() < 2) {
      return {0, 0};
    }

    auto rows = kernelMemRefType.getShape()[0];
    auto cols = kernelMemRefType.getShape()[1];
    if (rows <= 0 || cols <= 0) {
      return {0, 0};
    }

    return {static_cast<uint64_t>(rows), static_cast<uint64_t>(cols)};
  }

  if (auto kernelTypeAttr = mlir::dyn_cast<KernelType>(kernelOperand.getType())) {
    auto rows = kernelTypeAttr.getRows();
    auto cols = kernelTypeAttr.getCols();
    if (rows <= 0 || cols <= 0) {
      return {0, 0};
    }

    return {static_cast<uint64_t>(rows), static_cast<uint64_t>(cols)};
  }

  return {0, 0};
}

mlir::Value ConvolutionOp::initializeAccumulator(mlir::OpBuilder &builder, mlir::Location loc) {
  return createFloatConstant(builder, loc, 0.0);
}

mlir::Value ConvolutionOp::accumulate(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value currentAcc,
                                      mlir::Value pixelValue, mlir::Value optionalKernelValue) {
  auto img = pixelValue;
  auto kernelWeight = optionalKernelValue;
  (void)img;
  return builder
      .create<mlir::arith::AddFOp>(loc, currentAcc, builder.create<mlir::arith::MulFOp>(loc, pixelValue, kernelWeight))
      .getResult();
}

mlir::Value ConvolutionOp::finalizeAccumulator(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value finalAcc) {
  (void)builder;
  (void)loc;
  return finalAcc;
}

std::pair<uint64_t, uint64_t> ConvolutionOp::getNeighborhoodSize(mlir::ArrayRef<mlir::Value> operands) {
  if (operands.size() < 2) {
    return {0, 0};
  }

  [[maybe_unused]] auto img = operands[0];
  auto kernelOperand = operands[1];
  (void)img;

  return getKernelNeighborhoodSize(kernelOperand);
}

} // namespace picceler