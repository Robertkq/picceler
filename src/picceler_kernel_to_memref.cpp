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

  mlir::LogicalResult matchAndRewrite(KernelConstOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto f64Type = rewriter.getF64Type();

    // Get metadata from the custom type
    auto kernelType = mlir::cast<picceler::KernelType>(op.getResult().getType());
    int64_t rows = kernelType.getRows();
    int64_t cols = kernelType.getCols();

    // Create the memref type locally
    auto memrefType = mlir::MemRefType::get({rows * cols}, f64Type);

    // Allocation
    auto allocaOp = rewriter.create<mlir::memref::AllocaOp>(loc, memrefType);
    mlir::Value kStack = allocaOp.getResult();

    // Get the values attribute
    auto kvaluesAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op.getValuesAttr());
    if (!kvaluesAttr)
      return mlir::failure();

    // Store constants
    int64_t i = 0;
    for (double val : kvaluesAttr.getValues<double>()) {
      auto cIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i++);
      auto cVal = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(val));
      rewriter.create<mlir::memref::StoreOp>(loc, cVal, kStack, mlir::ValueRange{cIdx});
    }

    // Replace the old Op with the new MemRef
    rewriter.replaceOp(op, kStack);
    return mlir::success();
  }
};

void PiccelerKernelToMemrefPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<KernelToMemref>(&getContext());

  if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace picceler
