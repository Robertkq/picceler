#include "ops.h"
#include "types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <limits>

namespace picceler {

mlir::Value createFloatConstant(mlir::OpBuilder &builder, mlir::Location loc, double value) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, builder.getF64Type(), llvm::APFloat(value));
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

  auto img = operands[0];
  auto kernelOperand = operands[1];
  (void)img;

  return getKernelNeighborhoodSize(kernelOperand);
}

} // namespace picceler

#define GET_OP_INTERFACE_DEFS
#include "piccelerInterfaces.cpp.inc"
#undef GET_OP_INTERFACE_DEFS

#define GET_OP_CLASSES
#include "piccelerOps.cpp.inc"
#undef GET_OP_CLASSES