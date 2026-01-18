#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

#include "ops.h"
#include "types.h"

namespace picceler {

struct KernelToMemref : mlir::OpRewritePattern<KernelConstOp> {
  using OpRewritePattern<KernelConstOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(KernelConstOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto f64Type = rewriter.getF64Type();

    // Get metadata from the custom type
    auto kernelType = mlir::cast<picceler::KernelType>(op.getResult().getType());
    int64_t rows = kernelType.getRows();
    int64_t cols = kernelType.getCols();

    // Create a 2D memref type to preserve kernel dimensions
    auto memrefType = mlir::MemRefType::get({rows, cols}, f64Type);

    // Allocation
    auto allocaOp = rewriter.create<mlir::memref::AllocaOp>(loc, memrefType);
    mlir::Value kStack = allocaOp.getResult();

    // Get the values attribute
    auto kvaluesAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op.getValuesAttr());
    if (!kvaluesAttr) return mlir::failure();

    // Store constants in 2D layout
    int64_t i = 0;
    for (double val : kvaluesAttr.getValues<double>()) {
      int64_t row = i / cols;
      int64_t col = i % cols;
      auto cRowIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, row);
      auto cColIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, col);
      auto cVal = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(val));
      rewriter.create<mlir::memref::StoreOp>(loc, cVal, kStack, mlir::ValueRange{cRowIdx, cColIdx});
      i++;
    }

    // Replace the old Op with the new MemRef
    rewriter.replaceOp(op, kStack);
    return mlir::success();
  }
};

void PiccelerKernelToMemrefPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<KernelToMemref>(&getContext());

  // Apply this as a local pattern rewrite on existing operations using the
  // Greedy Pattern Rewriter, rather than setting up a full conversion target
  // and type-conversion pipeline for this simple, type-preserving transform.
  if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace picceler
