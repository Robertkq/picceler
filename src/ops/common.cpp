#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ops.h"

namespace picceler {

mlir::Value createFloatConstant(mlir::OpBuilder &builder, mlir::Location loc, double value) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, builder.getF64Type(), llvm::APFloat(value));
}

mlir::Value createIntConstant(mlir::OpBuilder &builder, mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64);
}

mlir::affine::AffineParallelOp createAffineParallel(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                                                    mlir::ValueRange upperBounds, mlir::TypeRange resultTypes,
                                                    llvm::ArrayRef<mlir::arith::AtomicRMWKind> reductions) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  unsigned n = upperBounds.size();

  llvm::SmallVector<mlir::AffineMap, 4> lbMaps(n, rewriter.getConstantAffineMap(0));
  llvm::SmallVector<mlir::AffineMap, 4> ubMaps;
  for (unsigned i = 0; i < n; ++i) {
    ubMaps.push_back(mlir::AffineMap::get(n, 0, rewriter.getAffineDimExpr(i), ctx));
  }
  llvm::SmallVector<int64_t, 4> steps(n, 1);

  return rewriter.create<mlir::affine::AffineParallelOp>(loc, resultTypes, reductions, lbMaps, mlir::ValueRange{},
                                                         ubMaps, upperBounds, steps);
}

} // namespace picceler