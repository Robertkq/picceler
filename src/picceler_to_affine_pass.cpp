#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"

#include <iostream>

#include "ops.h"
#include "types.h"
#include "image_access_helper.h"

namespace picceler {

struct BrightnessToAffine : mlir::OpConversionPattern<BrightnessOp> {
  using OpConversionPattern::OpConversionPattern;
  // Implementation of the pattern to convert BrightnessOp to Affine dialect
  mlir::LogicalResult matchAndRewrite(BrightnessOp op, BrightnessOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    mlir::Value input = adaptor.getInput();
    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    if (input.getType() != ptrType) {
      input = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, input).getResult(0);
    }

    ImageAccessHelper inputImage(input, rewriter, loc);
    mlir::Value inputWidthI32 = inputImage.getWidth();
    mlir::Value inputHeightI32 = inputImage.getHeight();
    mlir::Value inputDataPtr = inputImage.getDataPtr();

    auto createCall = rewriter.create<mlir::func::CallOp>(loc, ptrType, "piccelerCreateImage",
                                                          mlir::ValueRange{inputWidthI32, inputHeightI32});
    mlir::Value output = createCall.getResult(0);

    ImageAccessHelper outputImage(output, rewriter, loc);
    mlir::Value outputWidthI32 = outputImage.getWidth();
    mlir::Value outputHeightI32 = outputImage.getHeight();
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), inputWidthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), inputHeightI32);

    auto ubMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              height, ubMap, 1);

    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto colLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              width, ubMap, 1);

    rewriter.setInsertionPointToStart(colLoop.getBody());

    mlir::Value pixelRowIndex = rowLoop.getInductionVar();
    mlir::Value pixelColIndex = colLoop.getInductionVar();

    auto indexMap = mlir::AffineMap::get(
        2, 1, (rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) + rewriter.getAffineDimExpr(1)) * 4);

    mlir::Value pixelBaseIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, indexMap, mlir::ValueRange{pixelRowIndex, pixelColIndex, width});

    mlir::Value amount = adaptor.getValue();
    mlir::Value amountI32 = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI32Type(), amount);

    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    mlir::Value c255 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 255, 32);

    auto processChannel = [&](int offset, bool applyBrightness) {
      mlir::Value cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      mlir::Value byteAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, cOffset);
      mlir::Value byteAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), byteAddr);

      auto inputBytePtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, rewriter.getI8Type(), inputDataPtr, byteAddrI64);
      auto inputByte = rewriter.create<mlir::LLVM::LoadOp>(loc, rewriter.getI8Type(), inputBytePtr);

      mlir::Value finalValue;
      if (applyBrightness) {
        auto inputByteI32 = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI32Type(), inputByte);
        auto brightened = rewriter.create<mlir::arith::AddIOp>(loc, inputByteI32, amountI32);
        auto clampedLow = rewriter.create<mlir::arith::MinSIOp>(loc, brightened, c255);
        auto clampedHigh = rewriter.create<mlir::arith::MaxSIOp>(loc, clampedLow, c0);
        finalValue = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI8Type(), clampedHigh);

      } else {
        finalValue = inputByte;
      }

      auto outputBytePtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, rewriter.getI8Type(), outputDataPtr, byteAddrI64);
      auto outputByte = rewriter.create<mlir::LLVM::StoreOp>(loc, finalValue, outputBytePtr);
    };

    processChannel(0, true);  // R
    processChannel(1, true);  // G
    processChannel(2, true);  // B
    processChannel(3, false); // A

    // output
    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

struct InvertToAffine : mlir::OpConversionPattern<InvertOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(InvertOp op, InvertOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    mlir::Value input = adaptor.getInput();
    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    if (input.getType() != ptrType) {
      input = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, input).getResult(0);
    }

    ImageAccessHelper inputImage(input, rewriter, loc);
    mlir::Value inputWidthI32 = inputImage.getWidth();
    mlir::Value inputHeightI32 = inputImage.getHeight();
    mlir::Value inputDataPtr = inputImage.getDataPtr();

    auto createCall = rewriter.create<mlir::func::CallOp>(loc, ptrType, "piccelerCreateImage",
                                                          mlir::ValueRange{inputWidthI32, inputHeightI32});
    mlir::Value output = createCall.getResult(0);

    ImageAccessHelper outputImage(output, rewriter, loc);
    mlir::Value outputWidthI32 = outputImage.getWidth();
    mlir::Value outputHeightI32 = outputImage.getHeight();
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), inputWidthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getIndexType(), inputHeightI32);

    auto ubMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              height, ubMap, 1);

    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto colLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              width, ubMap, 1);

    rewriter.setInsertionPointToStart(colLoop.getBody());

    mlir::Value pixelRowIndex = rowLoop.getInductionVar();
    mlir::Value pixelColIndex = colLoop.getInductionVar();

    auto indexMap = mlir::AffineMap::get(
        2, 1, (rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) + rewriter.getAffineDimExpr(1)) * 4);

    mlir::Value pixelBaseIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, indexMap, mlir::ValueRange{pixelRowIndex, pixelColIndex, width});

    auto c255 = rewriter.create<mlir::arith::ConstantIntOp>(loc, rewriter.getI8Type(), 255);

    auto processChannel = [&](int offset, bool applyInvert) {
      mlir::Value cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      mlir::Value byteAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, cOffset);
      mlir::Value byteAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, rewriter.getI64Type(), byteAddr);

      auto inputBytePtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, rewriter.getI8Type(), inputDataPtr, byteAddrI64);
      auto inputByte = rewriter.create<mlir::LLVM::LoadOp>(loc, rewriter.getI8Type(), inputBytePtr);

      mlir::Value finalValue;
      if (applyInvert) {
        finalValue = rewriter.create<mlir::arith::SubIOp>(loc, c255, inputByte);
      } else {
        finalValue = inputByte;
      }

      auto outputBytePtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, rewriter.getI8Type(), outputDataPtr, byteAddrI64);
      auto outputByte = rewriter.create<mlir::LLVM::StoreOp>(loc, finalValue, outputBytePtr);
    };

    processChannel(0, true);  // R
    processChannel(1, true);  // G
    processChannel(2, true);  // B
    processChannel(3, false); // A

    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct ConvolutionToAffine : mlir::OpConversionPattern<ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(ConvolutionOp op, ConvolutionOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto indexType = rewriter.getIndexType();
    auto i8Type = rewriter.getI8Type();
    // auto f32Type = rewriter.getF32Type();
    auto f64Type = rewriter.getF64Type();
    auto i64Type = rewriter.getI64Type();
    // can or should i be using stuff like "using alias = ..." here?

    mlir::Location loc = op.getLoc();

    mlir::Value input = adaptor.getInput();
    mlir::Value kStack = adaptor.getKernel();

    auto kMemTy = mlir::dyn_cast<mlir::MemRefType>(kStack.getType());
    if (!kMemTy) {
      if (auto def = kStack.getDefiningOp()) {
        def->emitError("kernel operand must be lowered to a memref by PiccelerKernelToMemrefPass before ConvolutionToAffine");
      } else {
        op.emitError("kernel operand must be a memref");
      }
      return mlir::failure();
    }

    int64_t rows = 0, cols = 0;
    if (kMemTy.getRank() == 2) {
      rows = kMemTy.getShape()[0];
      cols = kMemTy.getShape()[1];
    } else {
      op.emitError("expected kernel memref of rank 2");
      return mlir::failure();
    }
    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    if (input.getType() != ptrType) {
      input = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, input).getResult(0);
    }

    ImageAccessHelper inputImage(input, rewriter, loc);
    mlir::Value inputWidthI32 = inputImage.getWidth();
    mlir::Value inputHeightI32 = inputImage.getHeight();
    mlir::Value inputDataPtr = inputImage.getDataPtr();

    auto createCall = rewriter.create<mlir::func::CallOp>(loc, ptrType, "piccelerCreateImage",
                                                          mlir::ValueRange{inputWidthI32, inputHeightI32});
    mlir::Value output = createCall.getResult(0);

    ImageAccessHelper outputImage(output, rewriter, loc);
    mlir::Value outputWidthI32 = outputImage.getWidth();
    mlir::Value outputHeightI32 = outputImage.getHeight();
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputWidthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputHeightI32);

    auto c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto v0 = rewriter.create<mlir::arith::ConstantIntOp>(loc, i8Type, 0);

    auto ubMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());

    auto oneConstant = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
    auto sumR = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, f64Type, oneConstant);
    auto sumG = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, f64Type, oneConstant);
    auto sumB = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, f64Type, oneConstant);

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              height, ubMap, 1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto pixelRowIndex = rowLoop.getInductionVar();
    auto coolLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{},
                                                               rewriter.getConstantAffineMap(0), width, ubMap, 1);
    rewriter.setInsertionPointToStart(coolLoop.getBody());

    auto c0_f32 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(0.0));
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0_f32, sumR);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0_f32, sumG);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0_f32, sumB);

    auto pixelColIndex = coolLoop.getInductionVar();

    auto indexMap = mlir::AffineMap::get(
        2, 1, (rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) + rewriter.getAffineDimExpr(1)) * 4);

    mlir::Value pixelBaseIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, indexMap, mlir::ValueRange{pixelRowIndex, pixelColIndex, width});

    auto kRowLoop = rewriter.create<mlir::affine::AffineForOp>(loc, 0, rows);

    rewriter.setInsertionPointToStart(kRowLoop.getBody());
    auto kx = kRowLoop.getInductionVar();

    auto kColLoop = rewriter.create<mlir::affine::AffineForOp>(loc, 0, cols);

    rewriter.setInsertionPointToStart(kColLoop.getBody());
    auto ky = kColLoop.getInductionVar();

    // d0: AxisIndex, d1: kAxisIndex, s0: half-width
    auto coordMap = mlir::AffineMap::get(
        2, 1, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1) - rewriter.getAffineSymbolExpr(0));

    mlir::Value rowOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, rows / 2);
    mlir::Value colOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, cols / 2);

    mlir::Value nRow =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, coordMap, mlir::ValueRange{pixelRowIndex, kx, rowOffset});
    mlir::Value nCol =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, coordMap, mlir::ValueRange{pixelColIndex, ky, colOffset});

    auto nPixelBaseIndex =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, indexMap, mlir::ValueRange{nRow, nCol, width});

    auto rowLow = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, nRow, c0);
    auto rowHigh = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nRow, height);

    auto colLow = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, nCol, c0);
    auto colHigh = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nCol, width);

    auto rowValid = rewriter.create<mlir::arith::AndIOp>(loc, rowLow, rowHigh);
    auto colValid = rewriter.create<mlir::arith::AndIOp>(loc, colLow, colHigh);
    auto isValid = rewriter.create<mlir::arith::AndIOp>(loc, rowValid, colValid);

    auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, isValid.getResult(), false);

    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    // Load the weight from stack-allocated 2D kernel memref
    auto kWeight = rewriter.create<mlir::memref::LoadOp>(loc, kStack, mlir::ValueRange{kx, ky});

    auto multiplyAndAccumulate = [&](int offset, mlir::Value sumAlloca, bool applyConvolution) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto nByteAddr = rewriter.create<mlir::arith::AddIOp>(loc, nPixelBaseIndex, cOffset);
      mlir::Value nByteAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, nByteAddr);

      auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, inputDataPtr, mlir::ValueRange{nByteAddrI64});
      auto byteVal = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, ptr);
      auto floatVal = rewriter.create<mlir::arith::UIToFPOp>(loc, f64Type, byteVal);

      auto currentSum = rewriter.create<mlir::LLVM::LoadOp>(loc, f64Type, sumAlloca);
      auto product = rewriter.create<mlir::arith::MulFOp>(loc, floatVal, kWeight);
      auto nextSum = rewriter.create<mlir::arith::AddFOp>(loc, currentSum, product);
      rewriter.create<mlir::LLVM::StoreOp>(loc, nextSum, sumAlloca);
    };

    multiplyAndAccumulate(0, sumR, true);
    multiplyAndAccumulate(1, sumG, true);
    multiplyAndAccumulate(2, sumB, true);

    rewriter.setInsertionPointAfter(kRowLoop);

    auto finalizePixel = [&](int offset, mlir::Value sumAlloca) {
      auto finalSum = rewriter.create<mlir::LLVM::LoadOp>(loc, f64Type, sumAlloca);

      auto c0_f64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(0.0));
      auto c255_f64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(255.0));
      auto clampedLow = rewriter.create<mlir::arith::MaximumFOp>(loc, finalSum, c0_f64);
      auto clampedFinal = rewriter.create<mlir::arith::MinimumFOp>(loc, clampedLow, c255_f64);

      auto byteVal = rewriter.create<mlir::arith::FPToUIOp>(loc, i8Type, clampedFinal);

      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto outAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, cOffset);
      auto outAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, outAddr);

      auto outPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{outAddrI64});

      rewriter.create<mlir::LLVM::StoreOp>(loc, byteVal, outPtr);
    };

    finalizePixel(0, sumR);
    finalizePixel(1, sumG);
    finalizePixel(2, sumB);

    // copy alpha channel
    auto c3 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 3);
    auto alphaAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, c3);
    auto alphaAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, alphaAddr);

    auto inAlphaPtr =
        rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, inputDataPtr, mlir::ValueRange{alphaAddrI64});
    auto alphaVal = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, inAlphaPtr);
    auto outAlphaPtr =
        rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{alphaAddrI64});
    rewriter.create<mlir::LLVM::StoreOp>(loc, alphaVal, outAlphaPtr);

    rewriter.setInsertionPointAfter(rowLoop);

    rewriter.replaceOp(op, output);

    return mlir::success();
  }
};

void PiccelerToAffinePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = &getContext();
  mlir::Location loc = module.getLoc();

  mlir::OpBuilder builder(&module.getBodyRegion());

  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i32Type = mlir::IntegerType::get(ctx, 32);

  if (!module.lookupSymbol<mlir::func::FuncOp>("piccelerCreateImage")) {
    auto funcType = builder.getFunctionType({i32Type, i32Type}, ptrType);
    auto func = builder.create<mlir::func::FuncOp>(loc, "piccelerCreateImage", funcType);
    func.setPrivate();
  }

  mlir::TypeConverter typeConverter;

  typeConverter.addConversion(
      [&](picceler::ImageType type) { return mlir::LLVM::LLVMPointerType::get(type.getContext()); });
  typeConverter.addConversion([](mlir::Type type) { return type; });

  typeConverter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });

  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                         mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();

  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addLegalOp<LoadImageOp, SaveImageOp, ShowImageOp, StringConstOp, KernelConstOp>();
  target.addIllegalOp<BrightnessOp, InvertOp, ConvolutionOp>();

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<BrightnessToAffine>(typeConverter, ctx);
  patterns.add<InvertToAffine>(typeConverter, ctx);
  patterns.add<ConvolutionToAffine>(typeConverter, ctx);

  if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    spdlog::error("Failed to convert Picceler operations to Affine dialect");
    signalPassFailure();
  }
}

} // namespace picceler