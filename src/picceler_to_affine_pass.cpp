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
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"

#include "ops.h"
#include "types.h"
#include "dialect.h"
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

struct RotateToAffine : mlir::OpConversionPattern<RotateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(RotateOp op, RotateOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    auto i8Type = rewriter.getI8Type();
    auto i64Type = rewriter.getI64Type();
    auto indexType = rewriter.getIndexType();

    mlir::Value input = adaptor.getInput();
    if (input.getType() != ptrType) {
      input = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, input).getResult(0);
    }

    ImageAccessHelper inputImage(input, rewriter, loc);
    mlir::Value inputWidthI32 = inputImage.getWidth();
    mlir::Value inputHeightI32 = inputImage.getHeight();
    mlir::Value inputDataPtr = inputImage.getDataPtr();

    auto c90I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 90, 64);
    auto c180I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 180, 64);
    auto c270I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 270, 64);

    mlir::Value angle = adaptor.getAngle();
    mlir::APInt constantAngle;
    if (!mlir::matchPattern(angle, mlir::m_ConstantInt(&constantAngle))) {
      return op.emitOpError("picceler.rotate requires angle to be a compile-time constant"), mlir::failure();
    }
    int64_t constantAngleValue = constantAngle.getSExtValue();
    if ((constantAngleValue % 90) != 0) {
      return op.emitOpError("angle must be a multiple of 90 degrees"), mlir::failure();
    }

    // Normalize signed angles into [0, 360), e.g. -90 -> 270.
    int64_t normalizedAngleValue = ((constantAngleValue % 360) + 360) % 360;
    mlir::Value normalizedAngle = rewriter.create<mlir::arith::ConstantIntOp>(loc, normalizedAngleValue, 64);

    mlir::Value is90 =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, normalizedAngle, c90I64);
    mlir::Value is180 =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, normalizedAngle, c180I64);
    mlir::Value is270 =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, normalizedAngle, c270I64);
    mlir::Value is90Or270 = rewriter.create<mlir::arith::OrIOp>(loc, is90, is270);

    mlir::Value outputWidthI32 = rewriter.create<mlir::arith::SelectOp>(loc, is90Or270, inputHeightI32, inputWidthI32);
    mlir::Value outputHeightI32 = rewriter.create<mlir::arith::SelectOp>(loc, is90Or270, inputWidthI32, inputHeightI32);

    auto createCall = rewriter.create<mlir::func::CallOp>(loc, ptrType, "piccelerCreateImage",
                                                          mlir::ValueRange{outputWidthI32, outputHeightI32});
    mlir::Value output = createCall.getResult(0);

    ImageAccessHelper outputImage(output, rewriter, loc);
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value inputWidth = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputWidthI32);
    mlir::Value inputHeight = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputHeightI32);
    mlir::Value outputWidth = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, outputWidthI32);
    mlir::Value outputHeight = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, outputHeightI32);

    auto ubMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              outputHeight, ubMap, 1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto colLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              outputWidth, ubMap, 1);
    rewriter.setInsertionPointToStart(colLoop.getBody());

    mlir::Value pixelRowIndex = rowLoop.getInductionVar();
    mlir::Value pixelColIndex = colLoop.getInductionVar();

    auto c1Index = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

    mlir::Value inputWidthMinusOne = rewriter.create<mlir::arith::SubIOp>(loc, inputWidth, c1Index);
    mlir::Value inputHeightMinusOne = rewriter.create<mlir::arith::SubIOp>(loc, inputHeight, c1Index);
    mlir::Value outputWidthMinusOne = rewriter.create<mlir::arith::SubIOp>(loc, outputWidth, c1Index);
    mlir::Value outputHeightMinusOne = rewriter.create<mlir::arith::SubIOp>(loc, outputHeight, c1Index);

    mlir::Value srcRowFor90DegRotation = pixelColIndex;
    mlir::Value srcColFor90DegRotation = rewriter.create<mlir::arith::SubIOp>(loc, outputHeightMinusOne, pixelRowIndex);
    mlir::Value srcRowFor180DegRotation = rewriter.create<mlir::arith::SubIOp>(loc, inputHeightMinusOne, pixelRowIndex);
    mlir::Value srcColFor180DegRotation = rewriter.create<mlir::arith::SubIOp>(loc, inputWidthMinusOne, pixelColIndex);
    mlir::Value srcRowFor270DegRotation = rewriter.create<mlir::arith::SubIOp>(loc, outputWidthMinusOne, pixelColIndex);
    mlir::Value srcColFor270DegRotation = pixelRowIndex;

    mlir::Value srcRow = rewriter.create<mlir::arith::SelectOp>(loc, is90, srcRowFor90DegRotation, pixelRowIndex);
    mlir::Value srcCol = rewriter.create<mlir::arith::SelectOp>(loc, is90, srcColFor90DegRotation, pixelColIndex);
    srcRow = rewriter.create<mlir::arith::SelectOp>(loc, is180, srcRowFor180DegRotation, srcRow);
    srcCol = rewriter.create<mlir::arith::SelectOp>(loc, is180, srcColFor180DegRotation, srcCol);
    srcRow = rewriter.create<mlir::arith::SelectOp>(loc, is270, srcRowFor270DegRotation, srcRow);
    srcCol = rewriter.create<mlir::arith::SelectOp>(loc, is270, srcColFor270DegRotation, srcCol);

    auto indexMap = mlir::AffineMap::get(
        2, 1, (rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) + rewriter.getAffineDimExpr(1)) * 4);

    mlir::Value dstPixelBaseIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, indexMap, mlir::ValueRange{pixelRowIndex, pixelColIndex, outputWidth});
    mlir::Value srcPixelBaseIndex =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, indexMap, mlir::ValueRange{srcRow, srcCol, inputWidth});

    auto copyChannel = [&](int offset) {
      auto srcOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto srcAddr = rewriter.create<mlir::arith::AddIOp>(loc, srcPixelBaseIndex, srcOffset);
      auto srcAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, srcAddr);
      auto srcPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, inputDataPtr, mlir::ValueRange{srcAddrI64});
      auto srcVal = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, srcPtr);

      auto dstOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto dstAddr = rewriter.create<mlir::arith::AddIOp>(loc, dstPixelBaseIndex, dstOffset);
      auto dstAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, dstAddr);
      auto dstPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{dstAddrI64});
      rewriter.create<mlir::LLVM::StoreOp>(loc, srcVal, dstPtr);
    };

    copyChannel(0);
    copyChannel(1);
    copyChannel(2);
    copyChannel(3);

    rewriter.setInsertionPointAfter(rowLoop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

struct NeighbourhoodOpsToAffine : mlir::OpInterfaceConversionPattern<NeighbourhoodOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  mlir::LogicalResult matchAndRewrite(NeighbourhoodOpInterface op, mlir::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto *rawOp = op.getOperation();
    mlir::Location loc = rawOp->getLoc();

    if (operands.empty()) {
      rawOp->emitOpError("expected at least one operand");
      return mlir::failure();
    }

    auto img = operands[0];
    mlir::Value kernelOperand;
    if (operands.size() > 1) {
      kernelOperand = operands[1];
    }

    auto [neighborhoodRows, neighborhoodCols] = op.getNeighborhoodSize(operands);
    if (neighborhoodRows == 0 || neighborhoodCols == 0) {
      rawOp->emitOpError("unable to determine a valid neighborhood size");
      return mlir::failure();
    }

    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    auto indexType = rewriter.getIndexType();
    auto i8Type = rewriter.getI8Type();
    auto f64Type = rewriter.getF64Type();
    auto i64Type = rewriter.getI64Type();

    mlir::Value input = img;
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
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputWidthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputHeightI32);

    mlir::Value neighborhoodRowsValue =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(neighborhoodRows));
    mlir::Value neighborhoodColsValue =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(neighborhoodCols));
    mlir::Value neighborhoodRowRadiusValue =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(neighborhoodRows / 2));
    mlir::Value neighborhoodColRadiusValue =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(neighborhoodCols / 2));
    mlir::Value zeroIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

    auto oneConstant = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
    auto sumR = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, f64Type, oneConstant);
    auto sumG = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, f64Type, oneConstant);
    auto sumB = rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, f64Type, oneConstant);

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0), height,
        mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()), 1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto colLoop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0), width,
        mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()), 1);
    rewriter.setInsertionPointToStart(colLoop.getBody());

    mlir::Value pixelRowIndex = rowLoop.getInductionVar();
    mlir::Value pixelColIndex = colLoop.getInductionVar();

    auto indexMap = mlir::AffineMap::get(
        2, 1, (rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) + rewriter.getAffineDimExpr(1)) * 4);

    mlir::Value pixelBaseIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, indexMap, mlir::ValueRange{pixelRowIndex, pixelColIndex, width});

    mlir::Value initAcc = op.initializeAccumulator(rewriter, loc);
    rewriter.create<mlir::LLVM::StoreOp>(loc, initAcc, sumR);
    rewriter.create<mlir::LLVM::StoreOp>(loc, initAcc, sumG);
    rewriter.create<mlir::LLVM::StoreOp>(loc, initAcc, sumB);

    auto kRowLoop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0), neighborhoodRowsValue,
        mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()), 1);
    rewriter.setInsertionPointToStart(kRowLoop.getBody());

    auto kColLoop = rewriter.create<mlir::affine::AffineForOp>(
        loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0), neighborhoodColsValue,
        mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext()), 1);
    rewriter.setInsertionPointToStart(kColLoop.getBody());

    mlir::Value kRowIndex = kRowLoop.getInductionVar();
    mlir::Value kColIndex = kColLoop.getInductionVar();

    auto coordMap = mlir::AffineMap::get(
        2, 1, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1) - rewriter.getAffineSymbolExpr(0));

    mlir::Value sampleRow = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, coordMap, mlir::ValueRange{pixelRowIndex, kRowIndex, neighborhoodRowRadiusValue});
    mlir::Value sampleCol = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, coordMap, mlir::ValueRange{pixelColIndex, kColIndex, neighborhoodColRadiusValue});

    auto rowLow = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, sampleRow, zeroIndex);
    auto rowHigh = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, sampleRow, height);
    auto colLow = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, sampleCol, zeroIndex);
    auto colHigh = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, sampleCol, width);

    auto rowValid = rewriter.create<mlir::arith::AndIOp>(loc, rowLow, rowHigh);
    auto colValid = rewriter.create<mlir::arith::AndIOp>(loc, colLow, colHigh);
    auto isValid = rewriter.create<mlir::arith::AndIOp>(loc, rowValid, colValid);

    auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, isValid.getResult(), false);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    mlir::Value sampleBaseIndex =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, indexMap, mlir::ValueRange{sampleRow, sampleCol, width});

    mlir::Value kernelWeight = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(1.0));
    if (kernelOperand && mlir::isa<mlir::MemRefType>(kernelOperand.getType())) {
      kernelWeight = rewriter.create<mlir::memref::LoadOp>(loc, kernelOperand, mlir::ValueRange{kRowIndex, kColIndex});
    }

    auto accumulateChannel = [&](int offset, mlir::Value sumAlloca) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto byteAddr = rewriter.create<mlir::arith::AddIOp>(loc, sampleBaseIndex, cOffset);
      auto byteAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, byteAddr);

      auto pixelPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, inputDataPtr, mlir::ValueRange{byteAddrI64});
      auto pixelByte = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, pixelPtr);
      auto pixelAsF64 = rewriter.create<mlir::arith::UIToFPOp>(loc, f64Type, pixelByte);

      auto currentAcc = rewriter.create<mlir::LLVM::LoadOp>(loc, f64Type, sumAlloca);
      auto nextAcc = op.accumulate(rewriter, loc, currentAcc, pixelAsF64, kernelWeight);
      rewriter.create<mlir::LLVM::StoreOp>(loc, nextAcc, sumAlloca);
    };

    accumulateChannel(0, sumR);
    accumulateChannel(1, sumG);
    accumulateChannel(2, sumB);

    rewriter.setInsertionPointAfter(kRowLoop);

    auto finalizeChannel = [&](int offset, mlir::Value sumAlloca) {
      auto currentAcc = rewriter.create<mlir::LLVM::LoadOp>(loc, f64Type, sumAlloca);
      auto finalizedAcc = op.finalizeAccumulator(rewriter, loc, currentAcc);

      auto c0F64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(0.0));
      auto c255F64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(255.0));
      auto clampedLow = rewriter.create<mlir::arith::MaximumFOp>(loc, finalizedAcc, c0F64);
      auto clampedHigh = rewriter.create<mlir::arith::MinimumFOp>(loc, clampedLow, c255F64);
      auto byteVal = rewriter.create<mlir::arith::FPToUIOp>(loc, i8Type, clampedHigh);

      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto outAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, cOffset);
      auto outAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, outAddr);
      auto outPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{outAddrI64});
      rewriter.create<mlir::LLVM::StoreOp>(loc, byteVal, outPtr);
    };

    finalizeChannel(0, sumR);
    finalizeChannel(1, sumG);
    finalizeChannel(2, sumB);

    auto c3 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 3);
    auto alphaAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, c3);
    auto alphaAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, alphaAddr);
    auto inputAlphaPtr =
        rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, inputDataPtr, mlir::ValueRange{alphaAddrI64});
    auto alphaVal = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, inputAlphaPtr);
    auto outputAlphaPtr =
        rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{alphaAddrI64});
    rewriter.create<mlir::LLVM::StoreOp>(loc, alphaVal, outputAlphaPtr);

    rewriter.setInsertionPointAfter(rowLoop);
    rewriter.replaceOp(rawOp, output);
    return mlir::success();
  }
};

struct ElementWiseBinaryOpToAffine : mlir::OpInterfaceConversionPattern<ElementWiseBinaryOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  mlir::LogicalResult matchAndRewrite(ElementWiseBinaryOpInterface op, mlir::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto *rawOp = op.getOperation();
    mlir::Location loc = rawOp->getLoc();

    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    auto i8Type = rewriter.getI8Type();
    auto i64Type = rewriter.getI64Type();
    auto indexType = rewriter.getIndexType();

    auto lhsImage = operands[0];
    auto rhsImage = operands[1];

    mlir::Value lhsInput = lhsImage;
    if (lhsInput.getType() != ptrType) {
      lhsInput = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, lhsInput).getResult(0);
    }

    mlir::Value rhsInput = rhsImage;
    if (rhsInput.getType() != ptrType) {
      rhsInput = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, rhsInput).getResult(0);
    }

    ImageAccessHelper lhsInputImage(lhsInput, rewriter, loc);
    mlir::Value lhsWidthI32 = lhsInputImage.getWidth();
    mlir::Value lhsHeightI32 = lhsInputImage.getHeight();
    mlir::Value lhsDataPtr = lhsInputImage.getDataPtr();

    ImageAccessHelper rhsInputImage(rhsInput, rewriter, loc);
    mlir::Value rhsWidthI32 = rhsInputImage.getWidth();
    mlir::Value rhsHeightI32 = rhsInputImage.getHeight();
    mlir::Value rhsDataPtr = rhsInputImage.getDataPtr();

    // compare dimensions and abort if they don't match
    auto widthMismatch =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhsWidthI32, rhsWidthI32);
    auto heightMismatch =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhsHeightI32, rhsHeightI32);
    auto dimMismatch = rewriter.create<mlir::arith::OrIOp>(loc, widthMismatch, heightMismatch);
    auto ifDimMismatch = rewriter.create<mlir::scf::IfOp>(loc, dimMismatch.getResult(), false);
    rewriter.setInsertionPointToStart(ifDimMismatch.thenBlock());
    // abort if dimensions don't match
    rewriter.create<mlir::func::CallOp>(loc, "abort", mlir::TypeRange{}, mlir::ValueRange{});

    rewriter.setInsertionPointAfter(ifDimMismatch);

    auto createCall = rewriter.create<mlir::func::CallOp>(loc, ptrType, "piccelerCreateImage",
                                                          mlir::ValueRange{lhsWidthI32, lhsHeightI32});
    mlir::Value output = createCall.getResult(0);
    ImageAccessHelper outputImage(output, rewriter, loc);
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value width = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, lhsWidthI32);
    mlir::Value height = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, lhsHeightI32);

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

    auto pixelBaseIndex = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, indexMap, mlir::ValueRange{pixelRowIndex, pixelColIndex, width});

    auto processChannel = [&](int offset) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);
      auto byteAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, cOffset);
      auto byteAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, byteAddr);

      auto lhsBytePtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, lhsDataPtr, mlir::ValueRange{byteAddrI64});
      auto lhsByte = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, lhsBytePtr);

      auto rhsBytePtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, rhsDataPtr, mlir::ValueRange{byteAddrI64});
      auto rhsByte = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, rhsBytePtr);

      auto resultByte = op.transformPixels(rewriter, loc, lhsByte, rhsByte, offset);

      auto outAddr = rewriter.create<mlir::arith::AddIOp>(loc, pixelBaseIndex, cOffset);
      auto outAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, outAddr);
      auto outPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{outAddrI64});
      rewriter.create<mlir::LLVM::StoreOp>(loc, resultByte, outPtr);
    };

    processChannel(0);
    processChannel(1);
    processChannel(2);
    processChannel(3);

    rewriter.setInsertionPointAfter(rowLoop);
    rewriter.replaceOp(rawOp, output);

    return mlir::success();
  }
};

struct CropToAffine : mlir::OpConversionPattern<CropOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(CropOp op, CropOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto ptrType = mlir::LLVM::LLVMPointerType::get(getContext());
    auto i8Type = rewriter.getI8Type();
    auto i64Type = rewriter.getI64Type();
    auto indexType = rewriter.getIndexType();

    mlir::Value input = adaptor.getInput();
    if (input.getType() != ptrType) {
      input = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, ptrType, input).getResult(0);
    }

    ImageAccessHelper inputImage(input, rewriter, loc);
    mlir::Value inputWidthI32 = inputImage.getWidth();
    mlir::Value inputHeightI32 = inputImage.getHeight();
    mlir::Value inputDataPtr = inputImage.getDataPtr();

    mlir::Value xI32 = adaptor.getX();
    mlir::Value yI32 = adaptor.getY();
    mlir::Value cropW_I32 = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI32Type(), adaptor.getWidth());
    mlir::Value cropH_I32 = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI32Type(), adaptor.getHeight());

    // create output image with crop size
    auto createCall = rewriter.create<mlir::func::CallOp>(loc, ptrType, "piccelerCreateImage",
                                                          mlir::ValueRange{cropW_I32, cropH_I32});
    mlir::Value output = createCall.getResult(0);

    ImageAccessHelper outputImage(output, rewriter, loc);
    mlir::Value outputDataPtr = outputImage.getDataPtr();

    mlir::Value inputWidth = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputWidthI32);
    mlir::Value inputHeight = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, inputHeightI32);
    (void)inputWidth;
    (void)inputHeight;

    mlir::Value xIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, xI32);
    mlir::Value yIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, yI32);
    mlir::Value cropW = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, cropW_I32);
    mlir::Value cropH = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, cropH_I32);

    auto ubMap = mlir::AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());

    auto rowLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              cropH, ubMap, 1);
    rewriter.setInsertionPointToStart(rowLoop.getBody());

    auto colLoop = rewriter.create<mlir::affine::AffineForOp>(loc, mlir::ValueRange{}, rewriter.getConstantAffineMap(0),
                                                              cropW, ubMap, 1);
    rewriter.setInsertionPointToStart(colLoop.getBody());

    mlir::Value outRow = rowLoop.getInductionVar();
    mlir::Value outCol = colLoop.getInductionVar();

    // map output pixel (outRow, outCol) to input pixel (srcRow, srcCol)
    mlir::Value srcRow = rewriter.create<mlir::arith::AddIOp>(loc, yIndex, outRow);
    mlir::Value srcCol = rewriter.create<mlir::arith::AddIOp>(loc, xIndex, outCol);

    // indexMap: (row, col, width) -> (row * width + col) * 4
    auto indexMap = mlir::AffineMap::get(
        2, 1, (rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) + rewriter.getAffineDimExpr(1)) * 4);

    mlir::Value dstPixelBaseIndex =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, indexMap, mlir::ValueRange{outRow, outCol, cropW});
    mlir::Value srcPixelBaseIndex =
        rewriter.create<mlir::affine::AffineApplyOp>(loc, indexMap, mlir::ValueRange{srcRow, srcCol, inputWidth});

    auto copyChannel = [&](int offset) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, offset);

      auto srcAddr = rewriter.create<mlir::arith::AddIOp>(loc, srcPixelBaseIndex, cOffset);
      auto srcAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, srcAddr);
      auto srcPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, inputDataPtr, mlir::ValueRange{srcAddrI64});
      auto srcVal = rewriter.create<mlir::LLVM::LoadOp>(loc, i8Type, srcPtr);

      auto dstAddr = rewriter.create<mlir::arith::AddIOp>(loc, dstPixelBaseIndex, cOffset);
      auto dstAddrI64 = rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, dstAddr);
      auto dstPtr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, i8Type, outputDataPtr, mlir::ValueRange{dstAddrI64});
      rewriter.create<mlir::LLVM::StoreOp>(loc, srcVal, dstPtr);
    };

    copyChannel(0);
    copyChannel(1);
    copyChannel(2);
    copyChannel(3);

    rewriter.setInsertionPointAfter(rowLoop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

#define GEN_PASS_DEF_PICCELERTOAFFINE
#include "piccelerPasses.h.inc"

/**
 * @brief A pass that converts Picceler operations to the Affine dialect. This is the crucial step where we lower from
 * our high-level image processing operations to a more explicit representation that can be further lowered to LLVM IR.
 * Each Picceler operation is matched and rewritten into one or more Affine loops that perform the equivalent
 * computation. This pass also handles type conversions and ensures that necessary runtime functions (like
 * piccelerCreateImage) are declared.
 */
struct PiccelerToAffinePass : public impl::PiccelerToAffineBase<PiccelerToAffinePass> {
  void runOnOperation() override {
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

    if (!module.lookupSymbol<mlir::func::FuncOp>("abort")) {
      auto abortType = builder.getFunctionType({}, {});
      auto func = builder.create<mlir::func::FuncOp>(loc, "abort", abortType);
      func.setPrivate();
    }

    mlir::TypeConverter typeConverter;

    typeConverter.addConversion(
        [&](picceler::ImageType type) { return mlir::LLVM::LLVMPointerType::get(type.getContext()); });
    typeConverter.addConversion([](mlir::Type type) { return type; });

    typeConverter.addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType, mlir::ValueRange inputs,
                                               mlir::Location loc) -> mlir::Value {
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    });

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                           mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();

    target.addLegalOp<mlir::UnrealizedConversionCastOp, StringConstOp>();
    target.addIllegalDialect<PiccelerDialect>();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<BrightnessToAffine>(typeConverter, ctx);
    patterns.add<InvertToAffine>(typeConverter, ctx);
    patterns.add<RotateToAffine>(typeConverter, ctx);
    patterns.add<NeighbourhoodOpsToAffine>(typeConverter, ctx);
    patterns.add<ElementWiseBinaryOpToAffine>(typeConverter, ctx);
    patterns.add<CropToAffine>(typeConverter, ctx);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      spdlog::error("Failed to convert Picceler operations to Affine dialect");
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerToAffinePass() { return std::make_unique<PiccelerToAffinePass>(); }

} // namespace picceler
