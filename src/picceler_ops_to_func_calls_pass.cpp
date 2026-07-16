#include "passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

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
struct LoadImageToCall : public mlir::OpConversionPattern<LoadImageOp> {
  using mlir::OpConversionPattern<LoadImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(LoadImageOp op, LoadImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerLoadImage", {stringType}, {imageType}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> args;
    args.push_back(adaptor.getFilename());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

/**
 * @brief Pattern to lower ShowImageOp to a function call.
 */
struct ShowImageToCall : public mlir::OpConversionPattern<ShowImageOp> {
  using mlir::OpConversionPattern<ShowImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ShowImageOp op, ShowImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerShowImage", {imageType}, {}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 1> args;
    args.push_back(adaptor.getInput());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

/**
 * @brief Pattern to lower SaveImageOp to a function call.
 */
struct SaveImageToCall : public mlir::OpConversionPattern<SaveImageOp> {
  using OpConversionPattern<SaveImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(SaveImageOp op, SaveImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto imageType = ImageType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerSaveImage", {imageType, stringType}, {}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(adaptor.getInput());
    args.push_back(adaptor.getFilename());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

struct ReadNumberToCall : public mlir::OpConversionPattern<ReadNumberOp> {
  using OpConversionPattern<ReadNumberOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ReadNumberOp op, ReadNumberOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto f64Type = rewriter.getF64Type();

    auto func = ensureRuntimeFunc(module, "piccelerReadNumber", {stringType}, {f64Type}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(adaptor.getPrompt());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

struct ReadStringToCall : mlir::OpConversionPattern<ReadStringOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ReadStringOp op, ReadStringOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerReadString", {stringType}, {stringType}, rewriter, loc);
    llvm::SmallVector<mlir::Value, 2> args;
    args.push_back(adaptor.getPrompt());

    auto call = rewriter.create<mlir::func::CallOp>(loc, func, args);
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

#define GEN_PASS_DEF_PICCELEROPSTOFUNCCALLS
#include "piccelerPasses.h.inc"

/**
 * @brief A pass that converts image operations (like load, show, save) into function calls.
 */
struct PiccelerOpsToFuncCallsPass : public impl::PiccelerOpsToFuncCallsBase<PiccelerOpsToFuncCallsPass> {
  using PiccelerOpsToFuncCallsBase::PiccelerOpsToFuncCallsBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::ConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addIllegalOp<LoadImageOp, ShowImageOp, SaveImageOp, ReadNumberOp, ReadStringOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LoadImageToCall, ShowImageToCall, SaveImageToCall, ReadNumberToCall, ReadStringToCall>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass() {
  return std::make_unique<PiccelerOpsToFuncCallsPass>();
}

} // namespace picceler