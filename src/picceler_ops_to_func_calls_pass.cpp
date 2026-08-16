#include "passes.h"

#include "spdlog/spdlog.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "ops.h"
#include "types.h"
#include "image_access_helper.h"
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>

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
 * @brief Builds a memref<?x?x4xi8> LLVM descriptor directly from a raw data pointer and
 * dynamic height/width (both i64-typed, matching LLVMTypeConverter's default index type),
 * emitting a single UnrealizedConversionCastOp from the LLVM struct to the memref type.
 * This cast is a structural identity (LLVMTypeConverter converts memref<?x?x4xi8> to exactly
 * this struct type), so it folds away cleanly under ReconcileUnrealizedCastsPass, unlike a
 * cast straight from !llvm.ptr to memref.
 */
mlir::Value buildImageMemref(mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value dataPtr,
                             mlir::Value heightI64, mlir::Value widthI64) {
  auto kDynamic = mlir::ShapedType::kDynamic;
  auto targetMemRefType = mlir::MemRefType::get({kDynamic, kDynamic, 4}, rewriter.getIntegerType(8));

  mlir::LLVMTypeConverter llvmTypeConverter(rewriter.getContext());
  auto descTy = llvmTypeConverter.convertType(targetMemRefType);

  mlir::MemRefDescriptor desc = mlir::MemRefDescriptor::poison(rewriter, loc, descTy);

  auto c4 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 4, 64).getResult();
  auto rowStride = rewriter.create<mlir::arith::MulIOp>(loc, widthI64, c4).getResult();

  desc.setAllocatedPtr(rewriter, loc, dataPtr);
  desc.setAlignedPtr(rewriter, loc, dataPtr);
  desc.setConstantOffset(rewriter, loc, 0);
  desc.setSize(rewriter, loc, 0, heightI64);
  desc.setSize(rewriter, loc, 1, widthI64);
  desc.setConstantSize(rewriter, loc, 2, 4);
  desc.setStride(rewriter, loc, 0, rowStride);
  desc.setConstantStride(rewriter, loc, 1, 4);
  desc.setConstantStride(rewriter, loc, 2, 1);

  return rewriter.create<mlir::UnrealizedConversionCastOp>(loc, targetMemRefType, mlir::Value(desc)).getResult(0);
}

struct LoadImageToCall : public mlir::OpConversionPattern<LoadImageOp> {
  using mlir::OpConversionPattern<LoadImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(LoadImageOp op, LoadImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

    auto stringType = StringType::get(ctx);

    // void piccelerLoadImage(!picceler.string, !llvm.ptr, !llvm.ptr, !llvm.ptr)
    // args: filename, &data, &height, &width
    auto fn = ensureRuntimeFunc(module, "piccelerLoadImage",
                                /*inputs=*/{stringType, ptrTy, ptrTy, ptrTy},
                                /*results=*/{}, rewriter, loc);

    auto c1_i32 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 32).getResult();

    // LLVM alloca signature in your build expects:
    // (resultType, elementType, arraySize, alignment)
    auto dataSlot = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrTy, ptrTy, c1_i32, /*alignment=*/0);
    auto hSlot = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrTy, i64Ty, c1_i32, /*alignment=*/0);
    auto wSlot = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrTy, i64Ty, c1_i32, /*alignment=*/0);

    // call runtime
    rewriter.create<mlir::func::CallOp>(
        loc, fn, mlir::ValueRange{adaptor.getFilename(), dataSlot.getResult(), hSlot.getResult(), wSlot.getResult()});

    // loads return ops; take Value via getResult()
    auto dataPtr = rewriter.create<mlir::LLVM::LoadOp>(loc, ptrTy, dataSlot.getResult()).getResult();
    auto h64 = rewriter.create<mlir::LLVM::LoadOp>(loc, i64Ty, hSlot.getResult()).getResult();
    auto w64 = rewriter.create<mlir::LLVM::LoadOp>(loc, i64Ty, wSlot.getResult()).getResult();

    auto result = buildImageMemref(rewriter, loc, dataPtr, h64, w64);

    rewriter.replaceOp(op, result);
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

    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i32Type = rewriter.getI32Type();

    mlir::Value img = adaptor.getInput(); // memref<?x?x4xi8>

    auto ptrIdx = rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, img);
    auto ptrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), ptrIdx);
    auto dataPtr = rewriter.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, ptrI64.getResult());

    auto heightIdx = rewriter.create<mlir::memref::DimOp>(loc, img, 0);
    auto widthIdx = rewriter.create<mlir::memref::DimOp>(loc, img, 1);
    auto height32 = rewriter.create<mlir::arith::IndexCastOp>(loc, i32Type, heightIdx);
    auto width32 = rewriter.create<mlir::arith::IndexCastOp>(loc, i32Type, widthIdx);

    auto func = ensureRuntimeFunc(module, "piccelerShowImage", {ptrType, i32Type, i32Type}, {}, rewriter, loc);
    auto call = rewriter.create<mlir::func::CallOp>(loc, func, mlir::ValueRange{dataPtr, width32, height32});

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

    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i32Type = rewriter.getI32Type();
    auto stringType = StringType::get(ctx);

    mlir::Value img = adaptor.getInput(); // memref<?x?x4xi8>

    auto ptrIdx = rewriter.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, img);
    auto ptrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), ptrIdx);
    auto dataPtr = rewriter.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, ptrI64.getResult());

    auto height32 =
        rewriter.create<mlir::arith::IndexCastOp>(loc, i32Type, rewriter.create<mlir::memref::DimOp>(loc, img, 0));
    auto width32 =
        rewriter.create<mlir::arith::IndexCastOp>(loc, i32Type, rewriter.create<mlir::memref::DimOp>(loc, img, 1));

    auto func =
        ensureRuntimeFunc(module, "piccelerSaveImage", {ptrType, i32Type, i32Type, stringType}, {}, rewriter, loc);
    auto call = rewriter.create<mlir::func::CallOp>(
        loc, func, mlir::ValueRange{dataPtr, width32, height32, adaptor.getFilename()});

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

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion([](ImageType imageType) -> mlir::Type {
      auto kDynamic = mlir::ShapedType::kDynamic;
      return mlir::MemRefType::get({kDynamic, kDynamic, 4}, mlir::IntegerType::get(imageType.getContext(), 8));
    });

    mlir::ConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp, StringConstOp>();
    target.addLegalDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                           mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
    target.addIllegalOp<LoadImageOp, ShowImageOp, SaveImageOp, ReadNumberOp, ReadStringOp, PrintOp>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });
    target.addDynamicallyLegalOp<mlir::func::CallOp>([&](mlir::func::CallOp op) {
      return typeConverter.isLegal(op.getOperandTypes()) && typeConverter.isLegal(op.getResultTypes());
    });

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) { return typeConverter.isLegal(op); });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LoadImageToCall, ShowImageToCall, SaveImageToCall, ReadNumberToCall, ReadStringToCall, PrintToCalls>(
        typeConverter, &getContext());
    patterns.add<GenericTypeUpdatePattern>(typeConverter, &getContext());
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass() {
  return std::make_unique<PiccelerOpsToFuncCallsPass>();
}

} // namespace picceler