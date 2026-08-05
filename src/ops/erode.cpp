#include "ops.h"
#include "spdlog/spdlog.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <limits>

namespace picceler {

/**
 * @brief Verifies the ErodeOp to ensure that the radius is a non-negative integer if it's a constant.
 */
mlir::LogicalResult ErodeOp::verify() {

  auto radius = getRadius();

  auto producer = radius.getDefiningOp();

  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(producer)) {
    auto valueAttr = constant.getValue();
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
      int64_t value = intAttr.getInt();
      if (value < 0) {
        // TODO: can we print location of the ErodeOp that has this issue / can we report the location of the constant
        // that is given to this?
        spdlog::error("ErodeOp expects constant integers in range (0, n) negative integer given. ");

        return mlir::failure();
      }
    } else {
      spdlog::error("Radius of ErodeOp is non-integer constant, not allowed!");
      return mlir::failure();
    }
  } else {
    // value of radius is given as runtime integer, cannot verify this
  }

  return mlir::success();
}

/**
 * @brief Registers canonicalization patterns for the ErodeOp.
 * No canonicalization patterns
 */
void ErodeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &, mlir::MLIRContext *) {}

/**
 * @brief Folds the ErodeOp if the radius is a constant integer equal to 0.
 * In this case, the ErodeOp can be replaced with its input image, as eroding with a radius of 0 has no effect.
 *
 * @param adaptor The FoldAdaptor providing access to the operands and attributes of the ErodeOp.
 * @return mlir::OpFoldResult The folded result, which is the input image if the radius is 0, or an empty result if no
 * folding is possible.
 */
mlir::OpFoldResult ErodeOp::fold(FoldAdaptor adaptor) {
  if (auto radiusAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(adaptor.getRadius())) {
    auto radiusValue = radiusAttr.getInt();
    if (radiusValue == 0) {
      return getInput();
    }
  }
  return {};
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

Result<std::pair<mlir::Value, mlir::Value>> ErodeOp::getNeighborhoodSize(mlir::OpBuilder &builder, mlir::Location loc,
                                                                         mlir::ArrayRef<mlir::Value> operands) {
  if (operands.size() < 2) {
    return std::unexpected(CompileError("ErodeOp requires at least 2 operands: input image and radius"));
  }

  [[maybe_unused]] auto img = operands[0];
  auto radius = operands[1];

  auto doubleRadius = builder.create<mlir::arith::AddIOp>(loc, radius, radius);
  auto neighborhoodSize = builder.create<mlir::arith::AddIOp>(loc, doubleRadius, createIntConstant(builder, loc, 1));
  return std::make_pair(neighborhoodSize, neighborhoodSize);
}

} // namespace picceler