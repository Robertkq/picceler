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

Result<std::pair<mlir::Value, mlir::Value>> getKernelNeighborhoodSize(mlir::OpBuilder &builder, mlir::Location loc,
                                                                      mlir::Value kernelOperand) {
  if (auto kernelMemRefType = mlir::dyn_cast<mlir::MemRefType>(kernelOperand.getType())) {
    if (kernelMemRefType.getRank() < 2) {
      return std::unexpected(CompileError("Kernel operand must have rank of at least 2"));
    }

    auto rows = kernelMemRefType.getShape()[0];
    auto cols = kernelMemRefType.getShape()[1];
    if (rows <= 0 || cols <= 0) {
      return std::unexpected(CompileError("Invalid kernel dimensions"));
    }

    return std::make_pair(createIntConstant(builder, loc, rows), createIntConstant(builder, loc, cols));
  }

  if (auto kernelTypeAttr = mlir::dyn_cast<KernelType>(kernelOperand.getType())) {
    auto rows = kernelTypeAttr.getRows();
    auto cols = kernelTypeAttr.getCols();
    if (rows <= 0 || cols <= 0) {
      return std::unexpected(CompileError("Invalid kernel dimensions"));
    }

    return std::make_pair(createIntConstant(builder, loc, rows), createIntConstant(builder, loc, cols));
  }

  return std::unexpected(CompileError("Invalid kernel operand"));
}

mlir::Value ConvolutionOp::initializeAccumulator(mlir::OpBuilder &builder, mlir::Location loc) {
  return createFloatConstant(builder, loc, 0.0);
}

mlir::Value ConvolutionOp::accumulate(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value currentAcc,
                                      mlir::Value pixelValue, mlir::Value optionalKernelValue) {

  auto kernelWeight = optionalKernelValue;
  return builder
      .create<mlir::arith::AddFOp>(loc, currentAcc, builder.create<mlir::arith::MulFOp>(loc, pixelValue, kernelWeight))
      .getResult();
}

mlir::Value ConvolutionOp::finalizeAccumulator([[maybe_unused]] mlir::OpBuilder &builder,
                                               [[maybe_unused]] mlir::Location loc, mlir::Value finalAcc) {
  return finalAcc;
}

Result<std::pair<mlir::Value, mlir::Value>>
ConvolutionOp::getNeighborhoodSize(mlir::OpBuilder &builder, mlir::Location loc, mlir::ArrayRef<mlir::Value> operands) {
  if (operands.size() < 2) {
    return std::unexpected(CompileError("ConvolutionOp requires at least 2 operands: input image and kernel"));
  }

  [[maybe_unused]] auto img = operands[0];
  auto kernelOperand = operands[1];

  return getKernelNeighborhoodSize(builder, loc, kernelOperand);
}

} // namespace picceler