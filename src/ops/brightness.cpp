#include "ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <spdlog/spdlog.h>

namespace picceler {

mlir::LogicalResult BrightnessOp::verify() {
  llvm::APInt intVal;
  if (mlir::matchPattern(getValue(), mlir::m_ConstantInt(&intVal))) {
    int64_t val = intVal.getSExtValue();
    if (val < -255 || val > 255) {
      return emitOpError("brightness value must be in range [-255, 255], but got ") << val;
    }
  }

  return mlir::success();
}

struct ChainedBrightnessPattern : public mlir::OpRewritePattern<BrightnessOp> {
  using OpRewritePattern<BrightnessOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(BrightnessOp op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();

    auto prevBrightnessOp = input.getDefiningOp<BrightnessOp>();
    if (!prevBrightnessOp)
      return mlir::failure();

    if (!prevBrightnessOp.getResult().hasOneUse())
      return mlir::failure();

    llvm::APInt prevVal, currentVal;
    if (!mlir::matchPattern(prevBrightnessOp.getValue(), mlir::m_ConstantInt(&prevVal)) ||
        !mlir::matchPattern(op.getValue(), mlir::m_ConstantInt(&currentVal))) {
      return mlir::failure();
    }

    int64_t combinedValue = prevVal.getSExtValue() + currentVal.getSExtValue();

    mlir::Location loc = op.getLoc();
    auto newConstOp = rewriter.create<mlir::arith::ConstantIntOp>(loc, combinedValue, 64);

    rewriter.replaceOpWithNewOp<BrightnessOp>(op, op.getType(), prevBrightnessOp.getInput(), newConstOp);

    spdlog::info("ChainedBrightnessPattern applied: combined {} and {} into {}", prevVal.getSExtValue(),
                 currentVal.getSExtValue(), combinedValue);

    return mlir::success();
  }
};

void BrightnessOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<ChainedBrightnessPattern>(context);
}

mlir::OpFoldResult BrightnessOp::fold(FoldAdaptor adaptor) {
  auto valueAttr = adaptor.getValue();
  auto valueIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(valueAttr);
  if (!valueIntAttr) {
    return mlir::OpFoldResult();
  }
  auto value = valueIntAttr.getInt();
  if (value == 0) {
    return getInput();
  }
  return mlir::OpFoldResult();
}

} // namespace picceler
