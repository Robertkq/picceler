#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

#include "ops.h"
#include "types.h"

namespace picceler {

/**
 * @brief Ensures that a runtime function is declared in the module.
 * If the function does not exist, it is created.
 * @param module The MLIR module to check.
 * @param name The name of the function.
 * @param inputs The input types of the function.
 * @param results The result types of the function.
 * @param rewriter The pattern rewriter to use for creating the function.
 * @param loc The location to use for the function.
 * @return The function operation.
 */
mlir::func::FuncOp ensureRuntimeFunc(mlir::ModuleOp module, mlir::StringRef name, llvm::ArrayRef<mlir::Type> inputs,
                                     llvm::ArrayRef<mlir::Type> results, mlir::PatternRewriter &rewriter,
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

/**
 * @brief Pattern to lower LoadImageOp to a function call.
 */
struct LoadImageToCall : public mlir::OpRewritePattern<LoadImageOp> {
  using mlir::OpRewritePattern<LoadImageOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(LoadImageOp op, mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerLoadImage", {stringType}, {imageType}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> args;
    args.push_back(op.getFilename());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

/**
 * @brief Pattern to lower ShowImageOp to a function call.
 */
struct ShowImageToCall : public mlir::OpRewritePattern<ShowImageOp> {
  using mlir::OpRewritePattern<ShowImageOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ShowImageOp op, mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerShowImage", {imageType}, {}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> args;
    args.push_back(op.getInput());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

/**
 * @brief Pattern to lower SaveImageOp to a function call.
 */
struct SaveImageToCall : public mlir::OpRewritePattern<SaveImageOp> {
  using mlir::OpRewritePattern<SaveImageOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(SaveImageOp op, mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerSaveImage", {imageType, stringType}, {}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(op.getInput());
    args.push_back(op.getFilename());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

/**
 * @brief Pattern to lower BlurOp to a function call.
 */
struct BlurImageToCall : public mlir::OpRewritePattern<BlurOp> {
  using mlir::OpRewritePattern<BlurOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(BlurOp op, mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerBlurImage", {imageType, stringType}, {imageType}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(op.getInput());
    args.push_back(op.getMode());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

#define GEN_PASS_DEF_PICCELEROPSTOFUNCCALLS
#include "piccelerPasses.h.inc"

struct PiccelerOpsToFuncCallsPass : public impl::PiccelerOpsToFuncCallsBase<PiccelerOpsToFuncCallsPass> {
  using PiccelerOpsToFuncCallsBase::PiccelerOpsToFuncCallsBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LoadImageToCall, ShowImageToCall, SaveImageToCall, BlurImageToCall>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass() {
  return std::make_unique<PiccelerOpsToFuncCallsPass>();
}

} // namespace picceler