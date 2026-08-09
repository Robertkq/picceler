#include "passes.h"

#include "spdlog/spdlog.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "image_access_helper.h"
#include "ops.h"
#include "types.h"

namespace picceler {

//===----------------------------------------------------------------------===//
// PiccelerTypeConverter
//===----------------------------------------------------------------------===//

/**
 * @brief Converts Picceler dialect types to their lowered runtime
 * representations. Currently only `!picceler.image` -> `memref<?x?x4xi8>`
 * is non-trivial; everything else passes through unchanged.
 */
class PiccelerTypeConverter : public mlir::TypeConverter {
public:
  PiccelerTypeConverter() {
    // Identity fallback: anything we don't have a specific rule for
    // (StringType, f64, etc.) converts to itself.
    addConversion([](mlir::Type type) -> mlir::Type { return type; });

    // !picceler.image -> memref<?x?x4xi8>
    addConversion([](ImageType imageTy) -> mlir::Type {
      auto *ctx = imageTy.getContext();
      return mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, 4},
                                   mlir::IntegerType::get(ctx, 8));
    });

    // Source materialization: framework needs to convert a *converted*
    // value back to the original type (e.g. a memref being fed to an
    // op that still expects !picceler.image because it hasn't been
    // converted yet).
    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType, mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return nullptr;
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    });

    // Target materialization: framework needs to convert an
    // *unconverted* value into the target type expected by an already
    // legalized op.
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType, mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return nullptr;
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    });
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/**
 * @brief Ensures that a runtime function is declared in the module.
 * If the function does not exist, it is created.
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
 * @brief Given the opaque Image* handle returned by piccelerCreateImage,
 * uses ImageAccessHelper to read width/height/data directly out of the
 * struct (no separate accessor runtime calls needed), then packs them
 * into a value of `memrefType` via memref.reinterpret_cast.
 */
mlir::Value buildMemrefFromHandle(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, mlir::Value handle,
                                  mlir::MemRefType memrefType) {
  ImageAccessHelper access(handle, rewriter, loc);

  mlir::Value dataPtr = access.getDataPtr();  // !llvm.ptr
  mlir::Value widthI32 = access.getWidth();   // i32
  mlir::Value heightI32 = access.getHeight(); // i32

  mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), widthI32);
  mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), heightI32);

  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value four = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
  mlir::Value rowStride = rewriter.create<mlir::arith::MulIOp>(loc, width, four);

  // Bridge the raw !llvm.ptr into a 1-D memref so reinterpret_cast has a
  // memref-typed source. This unrealized cast is expected to resolve
  // during your LLVM lowering (or via a MemRefDescriptor pack if you
  // lower straight to LLVM here instead).
  mlir::Value baseMemref =
      rewriter
          .create<mlir::UnrealizedConversionCastOp>(
              loc, mlir::MemRefType::get({mlir::ShapedType::kDynamic}, rewriter.getI8Type()), dataPtr)
          .getResult(0);

  return rewriter.create<mlir::memref::ReinterpretCastOp>(
      loc, memrefType, baseMemref,
      /*offset=*/mlir::OpFoldResult(zero),
      /*sizes=*/llvm::SmallVector<mlir::OpFoldResult>{height, width, mlir::OpFoldResult(four)},
      /*strides=*/llvm::SmallVector<mlir::OpFoldResult>{rowStride, four, one});
}

/**
 * @brief Given a memref<?x?x4xi8> image, extracts the base pointer,
 * height, and width so they can be passed to runtime functions that
 * consume raw image data (piccelerShowImage / piccelerSaveImage).
 * These ops receive the raw triple directly, not the Image* struct
 * (that struct only exists transiently right after piccelerCreateImage).
 */
struct RawImageArgs {
  mlir::Value dataPtr;
  mlir::Value height;
  mlir::Value width;
};

RawImageArgs extractRawImageArgs(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, mlir::Value memrefVal) {
  auto *ctx = rewriter.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i32Type = rewriter.getI32Type();

  auto extractOp = rewriter.create<mlir::memref::ExtractStridedMetadataOp>(loc, memrefVal);

  mlir::Value baseMemref = extractOp.getBaseBuffer();
  mlir::Value height = extractOp.getSizes()[0];
  mlir::Value width = extractOp.getSizes()[1];

  mlir::Value dataPtr = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, baseMemref).getResult(0);
  mlir::Value heightI32 = rewriter.create<mlir::arith::IndexCastOp>(loc, i32Type, height);
  mlir::Value widthI32 = rewriter.create<mlir::arith::IndexCastOp>(loc, i32Type, width);

  return {dataPtr, heightI32, widthI32};
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/**
 * @brief Pattern to lower LoadImageOp to piccelerCreateImage + memref rebuild
 * via ImageAccessHelper.
 */
struct LoadImageToCall : public mlir::OpConversionPattern<LoadImageOp> {
  using mlir::OpConversionPattern<LoadImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(LoadImageOp op, LoadImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);

    auto resultType = getTypeConverter()->convertType(op.getType());
    auto memrefType = mlir::dyn_cast_or_null<mlir::MemRefType>(resultType);
    if (!memrefType)
      return rewriter.notifyMatchFailure(op, "failed to convert result to memref<?x?x4xi8>");

    auto func = ensureRuntimeFunc(module, "piccelerCreateImage", {stringType}, {ptrType}, rewriter, loc);
    mlir::Value handle =
        rewriter.create<mlir::func::CallOp>(loc, func, mlir::ValueRange{adaptor.getFilename()}).getResult(0);

    mlir::Value result = buildMemrefFromHandle(rewriter, loc, handle, memrefType);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

/**
 * @brief Pattern to lower ShowImageOp to a runtime call.
 */
struct ShowImageToCall : public mlir::OpConversionPattern<ShowImageOp> {
  using mlir::OpConversionPattern<ShowImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ShowImageOp op, ShowImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i32Type = rewriter.getI32Type();

    // adaptor.getInput() is already the converted memref<?x?x4xi8>.
    RawImageArgs args = extractRawImageArgs(rewriter, loc, adaptor.getInput());

    auto func = ensureRuntimeFunc(module, "piccelerShowImage", {ptrType, i32Type, i32Type}, {}, rewriter, loc);
    rewriter.create<mlir::func::CallOp>(loc, func, mlir::ValueRange{args.dataPtr, args.width, args.height});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/**
 * @brief Pattern to lower SaveImageOp to a runtime call.
 */
struct SaveImageToCall : public mlir::OpConversionPattern<SaveImageOp> {
  using OpConversionPattern<SaveImageOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(SaveImageOp op, SaveImageOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
    auto i32Type = rewriter.getI32Type();

    RawImageArgs args = extractRawImageArgs(rewriter, loc, adaptor.getInput());

    auto func =
        ensureRuntimeFunc(module, "piccelerSaveImage", {ptrType, i32Type, i32Type, stringType}, {}, rewriter, loc);
    rewriter.create<mlir::func::CallOp>(loc, func,
                                        mlir::ValueRange{args.dataPtr, args.width, args.height, adaptor.getFilename()});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReadNumberToCall : public mlir::OpConversionPattern<ReadNumberOp> {
  using OpConversionPattern<ReadNumberOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ReadNumberOp op, ReadNumberOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);
    auto f64Type = rewriter.getF64Type();

    auto func = ensureRuntimeFunc(module, "piccelerReadNumber", {stringType}, {f64Type}, rewriter, loc);
    auto call = rewriter.create<mlir::func::CallOp>(loc, func, mlir::ValueRange{adaptor.getPrompt()});
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

struct ReadStringToCall : mlir::OpConversionPattern<ReadStringOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ReadStringOp op, ReadStringOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto stringType = StringType::get(ctx);

    auto func = ensureRuntimeFunc(module, "piccelerReadString", {stringType}, {stringType}, rewriter, loc);
    auto call = rewriter.create<mlir::func::CallOp>(loc, func, mlir::ValueRange{adaptor.getPrompt()});
    rewriter.replaceOp(op, call.getResults());

    return mlir::success();
  }
};

struct PrintToCalls : mlir::OpConversionPattern<PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(PrintOp op, PrintOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

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

template <typename OpTy> class GenericOpTypeConversion : public mlir::OpConversionPattern<OpTy> {
public:
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

  mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 4> convertedResults;
    if (mlir::failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), convertedResults)))
      return rewriter.notifyMatchFailure(op, "result type conversion failed");

    mlir::OperationState state(op->getLoc(), op->getName().getStringRef(), adaptor.getOperands(), convertedResults,
                               op->getAttrs());
    for (auto &region : op->getRegions())
      state.addRegion();

    mlir::Operation *newOp = rewriter.create(state);
    for (auto regionPair : llvm::zip(op->getRegions(), newOp->getRegions()))
      rewriter.inlineRegionBefore(std::get<0>(regionPair), std::get<1>(regionPair), std::get<1>(regionPair).end());

    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_PICCELEROPSTOFUNCCALLS
#include "piccelerPasses.h.inc"

/**
 * @brief A pass that converts image operations (like load, show, save) and
 * I/O ops into runtime function calls, lowering !picceler.image to
 * memref<?x?x4xi8> in the process.
 */
struct PiccelerOpsToFuncCallsPass : public impl::PiccelerOpsToFuncCallsBase<PiccelerOpsToFuncCallsPass> {
  using PiccelerOpsToFuncCallsBase::PiccelerOpsToFuncCallsBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    PiccelerTypeConverter typeConverter;
    mlir::ConversionTarget target(getContext());

    target.addLegalOp<mlir::ModuleOp, StringConstOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                           mlir::LLVM::LLVMDialect>();

    target.addIllegalOp<LoadImageOp, ShowImageOp, SaveImageOp, ReadNumberOp, ReadStringOp, PrintOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LoadImageToCall, ShowImageToCall, SaveImageToCall, ReadNumberToCall, ReadStringToCall, PrintToCalls>(
        typeConverter, &getContext());

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerOpsToFuncCallsPass() {
  return std::make_unique<PiccelerOpsToFuncCallsPass>();
}

} // namespace picceler