#include <spdlog/spdlog.h>

#include "passes.h"
#include "ops.h"
#include "types.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>

namespace picceler {

mlir::func::FuncOp ensureRuntimeFunc(mlir::ModuleOp module,
                                     mlir::StringRef name,
                                     llvm::ArrayRef<mlir::Type> inputs,
                                     llvm::ArrayRef<mlir::Type> results,
                                     mlir::PatternRewriter &rewriter,
                                     mlir::Location loc) {
  auto func = module.lookupSymbol<mlir::func::FuncOp>(name);
  if (func)
    return func;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType = rewriter.getFunctionType(inputs, results);
  func = rewriter.create<mlir::func::FuncOp>(loc, name, funcType);
  func.setPrivate();
  return func;
}

struct LoadImageToCall : public mlir::OpRewritePattern<LoadImageOp> {
  using mlir::OpRewritePattern<LoadImageOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LoadImageOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "picceler_load_image", {stringType},
                                  {imageType}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> args;
    args.push_back(op.getFilename());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

struct ShowImageToCall : public mlir::OpRewritePattern<ShowImageOp> {
  using mlir::OpRewritePattern<ShowImageOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ShowImageOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "picceler_show_image", {imageType},
                                  {}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> args;
    args.push_back(op.getInput());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

struct SaveImageToCall : public mlir::OpRewritePattern<SaveImageOp> {
  using mlir::OpRewritePattern<SaveImageOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SaveImageOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "picceler_save_image",
                                  {imageType, stringType}, {}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(op.getInput());
    args.push_back(op.getFilename());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();

    return mlir::success();
  }
};

struct BlurImageToCall : public mlir::OpRewritePattern<BlurOp> {
  using mlir::OpRewritePattern<BlurOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(BlurOp op, mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func =
        ensureRuntimeFunc(module, "picceler_blur_image",
                          {imageType, stringType}, {imageType}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(op.getInput());
    args.push_back(op.getMode());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

void LowerPiccelerOpsToFuncCallsPass::runOnOperation() {
  spdlog::trace("Running LowerPiccelerOpsToFuncCallsPass");
  mlir::ModuleOp module = getOperation();

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<LoadImageToCall>(&getContext());
  patterns.add<BlurImageToCall>(&getContext());
  patterns.add<ShowImageToCall>(&getContext());
  patterns.add<SaveImageToCall>(&getContext());

  if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

mlir::StringRef LowerPiccelerOpsToFuncCallsPass::getArgument() const {
  return "lower-picceler-ops-to-func-calls";
}

mlir::StringRef LowerPiccelerOpsToFuncCallsPass::getDescription() const {
  return "Lower Picceler operations to function calls";
}

void LowerPiccelerOpsToFuncCallsPass::registerPass() {
  mlir::PassRegistration<LowerPiccelerOpsToFuncCallsPass>();
}

} // namespace picceler