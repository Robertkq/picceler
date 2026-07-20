#include "passes.h"

#include "spdlog/spdlog.h"

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

struct PrintToCalls : mlir::OpConversionPattern<PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);

    auto fmtValue = adaptor.getFmt();
    auto stringProducer = fmtValue.getDefiningOp<StringConstOp>();
    if (!stringProducer) {
      spdlog::error("PrintOp format value is not a StringConstOp");
      return mlir::failure();
    }
    auto args = adaptor.getArgs();

    std::string fmt = stringProducer.getValue().str();
    spdlog::debug("PrintOp format value: {}", fmt);

    std::vector<std::string> splitFmt = splitFormatString(fmt);
    size_t argsIndex = 0;
    for (const auto &part : splitFmt) {
      spdlog::debug("Split format part: {}", part);
      if (!part.empty()) {
        callFormatStringLiteral(rewriter, loc, part);
      }
      if (argsIndex < args.size()) {
        auto rawVal = args[argsIndex];
        auto type = rawVal.getType();
        if (auto floatVal = llvm::dyn_cast<mlir::TypedValue<mlir::Float64Type>>(rawVal)) {
          callFormatFloat64(rewriter, loc, floatVal);
        } else if (auto stringVal = llvm::dyn_cast<mlir::TypedValue<StringType>>(rawVal)) {
          callFormatString(rewriter, loc, stringVal);
        } else {
          op.emitOpError("Unsupported argument type for print format!");
          return mlir::failure();
        }
        argsIndex++;
      }
    }
    rewriter.eraseOp(stringProducer);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  void callFormatStringLiteral(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                               const std::string &str) const {
    auto stringType = StringType::get(rewriter.getContext());
    auto string = rewriter.create<StringConstOp>(loc, stringType, mlir::StringAttr::get(rewriter.getContext(), str));
    callFormatString(rewriter, loc, string.getResult());
  }
  void callFormatString(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                        mlir::TypedValue<StringType> value) const {
    auto module = rewriter.getInsertionBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    auto funcOp = ensureRuntimeFunc(module, "piccelerPrintString", {value.getType()}, {}, rewriter, loc);
    rewriter.create<mlir::func::CallOp>(loc, funcOp, mlir::ValueRange{value});
  }
  void callFormatFloat64(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                         mlir::TypedValue<mlir::Float64Type> value) const {
    auto module = rewriter.getInsertionBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();
    auto funcOp = ensureRuntimeFunc(module, "piccelerPrintFloat64", {value.getType()}, {}, rewriter, loc);
    rewriter.create<mlir::func::CallOp>(loc, funcOp, mlir::ValueRange{value});
  }

  std::vector<std::string> splitFormatString(const std::string &fmt) const {
    std::vector<std::string> parts;
    size_t start = 0;

    while (start < fmt.size()) {
      size_t pos = fmt.find("{}", start);
      if (pos == std::string::npos) {
        parts.push_back(fmt.substr(start));
        break;
      }

      parts.push_back(fmt.substr(start, pos - start));
      start = pos + 2;
    }
    return parts;
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
    target.addLegalOp<mlir::ModuleOp, StringConstOp>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addIllegalOp<LoadImageOp, ShowImageOp, SaveImageOp, ReadNumberOp, ReadStringOp, PrintOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LoadImageToCall, ShowImageToCall, SaveImageToCall, ReadNumberToCall, ReadStringToCall, PrintToCalls>(
        &getContext());

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass() {
  return std::make_unique<PiccelerOpsToFuncCallsPass>();
}

} // namespace picceler