#include "ops.h"
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include <spdlog/spdlog.h>

namespace picceler {

// TODO: Refactor SharpenOp to use strength as input instead of radius
//  and make it a fixed 3x3 kernel.

mlir::LogicalResult SharpenOp::verify() {

  if (getInput().getType() != getResult().getType()) {
    return emitOpError("input and result types must match");
  }

  return mlir::success();
}

void SharpenOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  // no canonicalization patterns for sharpen yet
  // to be implemented in the future when sharpen takes strength and is fixed 3x3 sliding window
}

mlir::OpFoldResult SharpenOp::fold(FoldAdaptor adaptor) {
  if (auto valueAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getValue())) {
    if (valueAttr.getInt() == 0) {
      return getInput(); // Fold away sharpen(img, 0) -> img
    }
  }
  return {};
}

} // namespace picceler