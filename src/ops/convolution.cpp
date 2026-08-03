#include "ops.h"
#include "types.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <spdlog/spdlog.h>

#include <algorithm>

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

/**
 * @brief Verifies the ConvolutionOp to ensure that the kernel dimensions are valid.
 * If the kernel is a KernelType, it checks that the rows and columns are positive.
 * If the kernel is a MemRefType, it checks that the rank is 2 and that the rows and columns are positive (if they are
 * not dynamic).
 *
 * @return mlir::success() if the verification passes, otherwise emits an error and returns mlir::failure().
 */
mlir::LogicalResult ConvolutionOp::verify() {
  auto kernelOperand = getKernel();
  if (auto kernelType = mlir::dyn_cast<KernelType>(kernelOperand.getType())) {
    if (kernelType.getRows() <= 0 || kernelType.getCols() <= 0) {
      return emitOpError("Kernel dimensions must be positive, got ")
             << kernelType.getRows() << "x" << kernelType.getCols();
    }
  } else if (auto kernelMemRefType = mlir::dyn_cast<mlir::MemRefType>(kernelOperand.getType())) {
    if (kernelMemRefType.getRank() != 2) {
      return emitOpError("Kernel operand must have rank 2, got ") << kernelMemRefType.getRank();
    }

    auto rows = kernelMemRefType.getShape()[0];
    if (!mlir::ShapedType::isDynamic(rows) && rows <= 0) {
      return emitOpError("Kernel rows must be positive, got ") << rows;
    }
    auto cols = kernelMemRefType.getShape()[1];
    if (!mlir::ShapedType::isDynamic(cols) && cols <= 0) {
      return emitOpError("Kernel cols must be positive, got ") << cols;
    }
  }

  return mlir::success();
}

struct IdentityConvolutionPattern : public mlir::OpRewritePattern<ConvolutionOp> {
  using mlir::OpRewritePattern<ConvolutionOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ConvolutionOp op, mlir::PatternRewriter &rewriter) const override {
    auto kernelOperand = op.getKernel();
    if (!mlir::isa<KernelType>(kernelOperand.getType())) {
      return mlir::failure();
    }

    auto kernelType = mlir::cast<KernelType>(kernelOperand.getType());
    auto rows = kernelType.getRows();
    auto cols = kernelType.getCols();

    if (rows != cols || rows % 2 == 0) {
      return mlir::failure();
    }

    auto definingOp = kernelOperand.getDefiningOp<KernelConstOp>();
    if (!definingOp) {
      return mlir::failure();
    }

    auto valuesAttr = definingOp.getValues();
    auto floatValues = valuesAttr.getValues<llvm::APFloat>();

    if (static_cast<int64_t>(floatValues.size()) != rows * cols) {
      return mlir::failure();
    }

    int64_t centerR = rows / 2;
    int64_t centerC = cols / 2;

    int64_t nonZeroCount = 0;

    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        double value = floatValues[i * cols + j].convertToDouble();

        if (value != 0.0) {
          ++nonZeroCount;
        }
      }
    }

    if (nonZeroCount != 1) {
      spdlog::debug("IdentityConvolutionPattern: Kernel has {} non-zero values, expected 1", nonZeroCount);
      return mlir::failure();
    }

    if (floatValues[centerR * cols + centerC].convertToDouble() != 1.0) {
      spdlog::debug("IdentityConvolutionPattern: Center value is not 1.0");
      return mlir::failure();
    }

    rewriter.replaceOp(op, op.getInput());
    return mlir::success();
  }
};

/**
 * @brief Registers canonicalization patterns for the ConvolutionOp.
 * Adds the IdentityConvolutionPattern to the provided set of rewrite patterns.
 * which performs convolution(img, kernel) -> img if the kernel is an identity kernel.
 *
 * @param results The set of rewrite patterns to which the canonicalization patterns will be added.
 * @param context The MLIR context in which the patterns are being registered.
 */
void ConvolutionOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<IdentityConvolutionPattern>(context);
}

mlir::OpFoldResult ConvolutionOp::fold(FoldAdaptor adaptor) { return {}; }

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