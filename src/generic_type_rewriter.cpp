#include "passes.h"

namespace picceler {

mlir::LogicalResult GenericTypeUpdatePattern::matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                                                              mlir::ConversionPatternRewriter &rewriter) const {

  if (getTypeConverter()->isLegal(op))
    return mlir::failure();

  mlir::SmallVector<mlir::Type> newResultTypes;
  if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(), newResultTypes)))
    return mlir::failure();

  mlir::OperationState state(op->getLoc(), op->getName().getStringRef(), operands, newResultTypes, op->getAttrs());
  mlir::Operation *newOp = rewriter.create(state);

  rewriter.replaceOp(op, newOp->getResults());
  return mlir::success();
}

} // namespace picceler
