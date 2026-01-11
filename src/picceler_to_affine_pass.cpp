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

#include "ops.h"
#include "types.h"
#include "image_access_helper.h"

namespace picceler {

struct BrightnessToAffine : mlir::OpConversionPattern<BrightnessOp> {
  using OpConversionPattern::OpConversionPattern;
  // Implementation of the pattern to convert BrightnessOp to Affine dialect
  mlir::LogicalResult
  matchAndRewrite(BrightnessOp op, BrightnessOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    mlir::Value input = adaptor.getInput();
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(input.getType())) {
      auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      input =
          rewriter
              .create<mlir::UnrealizedConversionCastOp>(loc, llvmPtrType, input)
              .getResult(0);
    }

    ImageAccessHelper imgHelper(input, rewriter, loc);
    mlir::Value widthI32 = imgHelper.getWidth();
    mlir::Value heightI32 = imgHelper.getHeight();
    mlir::Value dataPtr = imgHelper.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), widthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), heightI32);

    auto map = mlir::AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) * rewriter.getAffineDimExpr(1) * 4,
        rewriter.getContext());

    auto loop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
        mlir::ValueRange{width, height}, map, 4);

    rewriter.setInsertionPointToStart(loop.getBody());

    mlir::Value i = loop.getInductionVar();

    mlir::Value amount = adaptor.getValue();
    mlir::Value amountI32 = rewriter.create<mlir::arith::TruncIOp>(
        loc, rewriter.getI32Type(), amount);

    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
    mlir::Value c255 =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, 255, 32);

    auto processChannel = [&](int offset) {
      // Calculate pixel address
      mlir::Value offsetVal =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      mlir::Value currentIdx =
          rewriter.create<mlir::arith::AddIOp>(loc, i, offsetVal);

      mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getI64Type(), currentIdx);
      mlir::Value pixelPtr = rewriter.create<mlir::LLVM::GEPOp>(
          loc, mlir::LLVM::LLVMPointerType::get(getContext()),
          rewriter.getI8Type(), dataPtr, mlir::ValueRange{idxI64});

      mlir::Value pixelVal = rewriter.create<mlir::LLVM::LoadOp>(
          loc, rewriter.getI8Type(), pixelPtr);
      mlir::Value pixelValI32 = rewriter.create<mlir::arith::ExtUIOp>(
          loc, rewriter.getI32Type(), pixelVal);

      mlir::Value brightened =
          rewriter.create<mlir::arith::AddIOp>(loc, pixelValI32, amountI32);
      mlir::Value clampedLow =
          rewriter.create<mlir::arith::MaxSIOp>(loc, brightened, c0);
      mlir::Value clampedHigh =
          rewriter.create<mlir::arith::MinSIOp>(loc, clampedLow, c255);

      mlir::Value finalVal = rewriter.create<mlir::arith::TruncIOp>(
          loc, rewriter.getI8Type(), clampedHigh);
      rewriter.create<mlir::LLVM::StoreOp>(loc, finalVal, pixelPtr);
    };

    processChannel(0); // R
    processChannel(1); // G
    processChannel(2); // B

    rewriter.replaceOp(op, adaptor.getInput());

    return mlir::success();
  }
};

struct InvertToAffine : mlir::OpConversionPattern<InvertOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(InvertOp op, InvertOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    mlir::Value input = adaptor.getInput();
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(input.getType())) {
      auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      input =
          rewriter
              .create<mlir::UnrealizedConversionCastOp>(loc, llvmPtrType, input)
              .getResult(0);
    }

    ImageAccessHelper imgHelper(input, rewriter, loc);
    mlir::Value widthI32 = imgHelper.getWidth();
    mlir::Value heightI32 = imgHelper.getHeight();
    mlir::Value dataPtr = imgHelper.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), widthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), heightI32);

    auto map = mlir::AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) * rewriter.getAffineDimExpr(1) * 4,
        rewriter.getContext());

    auto loop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
        mlir::ValueRange{width, height}, map, 4);

    rewriter.setInsertionPointToStart(loop.getBody());

    mlir::Value i = loop.getInductionVar();

    auto processChannel = [&](int offset) {
      // Calculate pixel address
      mlir::Value offsetVal =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      mlir::Value currentIdx =
          rewriter.create<mlir::arith::AddIOp>(loc, i, offsetVal);

      mlir::Value idxI64 = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getI64Type(), currentIdx);
      mlir::Value pixelPtr = rewriter.create<mlir::LLVM::GEPOp>(
          loc, mlir::LLVM::LLVMPointerType::get(getContext()),
          rewriter.getI8Type(), dataPtr, mlir::ValueRange{idxI64});

      mlir::Value pixelVal = rewriter.create<mlir::LLVM::LoadOp>(
          loc, rewriter.getI8Type(), pixelPtr);

      mlir::Value inverted = rewriter.create<mlir::arith::SubIOp>(
          loc, rewriter.create<mlir::arith::ConstantIntOp>(loc, 255, 8),
          pixelVal);

      rewriter.create<mlir::LLVM::StoreOp>(loc, inverted, pixelPtr);
    };

    processChannel(0); // R
    processChannel(1); // G
    processChannel(2); // B
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

struct ConvolutionToAffine : mlir::OpConversionPattern<ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConvolutionOp op, ConvolutionOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Location loc = op.getLoc();

    mlir::Value input = adaptor.getInput();
    mlir::Value kernelValue = adaptor.getKernel();

    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(input.getType())) {
      auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(getContext());
      input =
          rewriter
              .create<mlir::UnrealizedConversionCastOp>(loc, llvmPtrType, input)
              .getResult(0);
    }

    ImageAccessHelper imgHelper(input, rewriter, loc);
    mlir::Value widthI32 = imgHelper.getWidth();
    mlir::Value heightI32 = imgHelper.getHeight();
    mlir::Value dataPtr = imgHelper.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), widthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), heightI32);

    auto rowMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0),
                                       rewriter.getContext());

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0), height,
        rowMap, 1);

    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto colMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0),
                                       rewriter.getContext());

    auto coolLoop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0), width,
        colMap, 1);

    rewriter.setInsertionPointToStart(coolLoop.getBody());

    auto rowIndex = rowLoop.getInductionVar();
    auto colIndex = coolLoop.getInductionVar();
    auto channels = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 4);
    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

    auto rowTimesWidth = rewriter.create<mlir::arith::MulIOp>(
        loc, rewriter.getIndexType(), rowIndex, width);
    auto rowPlusCol = rewriter.create<mlir::arith::AddIOp>(
        loc, rewriter.getIndexType(), rowTimesWidth, colIndex);
    auto flattenedBase = rewriter.create<mlir::arith::MulIOp>(
        loc, rewriter.getIndexType(), rowPlusCol, channels);

    return mlir::success();
  }
};

void PiccelerToAffinePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = &getContext();

  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();

  target.addLegalOp<mlir::UnrealizedConversionCastOp>();
  target.addLegalOp<LoadImageOp, SaveImageOp, ShowImageOp, StringConstOp>();

  target.addIllegalOp<BrightnessOp>();
  target.addIllegalOp<InvertOp>();
  target.addIllegalOp<ConvolutionOp>();

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<BrightnessToAffine>(ctx);
  patterns.add<InvertToAffine>(ctx);
  patterns.add<ConvolutionToAffine>(ctx);

  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    spdlog::error("Failed to convert Picceler operations to Affine dialect");
    signalPassFailure();
  }
}

} // namespace picceler