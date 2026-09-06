#include "passes.h"
#include "channels.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"

#include "ops.h"
#include "types.h"
#include "dialect.h"

namespace picceler {

struct RotateToAffine : mlir::OpConversionPattern<RotateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(RotateOp op, RotateOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto i8Type = rewriter.getI8Type();
    auto indexType = rewriter.getIndexType();

    mlir::Value input = adaptor.getInput();
    if (!mlir::isa<mlir::MemRefType>(input.getType())) {
      op.emitOpError("expected input image to be a MemRefType");
      return mlir::failure();
    }

    mlir::Value inputHeight = rewriter.create<mlir::memref::DimOp>(loc, input, 0);
    mlir::Value inputWidth = rewriter.create<mlir::memref::DimOp>(loc, input, 1);

    auto c90I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 90, 64);
    auto c180I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 180, 64);
    auto c270I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 270, 64);

    mlir::Value angle = adaptor.getAngle();
    mlir::Value normalizedAngle;
    mlir::APInt constantAngle;
    if (mlir::matchPattern(angle, mlir::m_ConstantInt(&constantAngle))) {
      int64_t constantAngleValue = constantAngle.getSExtValue();
      if ((constantAngleValue % 90) != 0) {
        return op.emitOpError("angle must be a multiple of 90 degrees"), mlir::failure();
      }

      // Normalize signed angles into [0, 360), e.g. -90 -> 270.
      int64_t normalizedAngleValue = ((constantAngleValue % 360) + 360) % 360;
      normalizedAngle = rewriter.create<mlir::arith::ConstantIntOp>(loc, normalizedAngleValue, 64);
    } else {
      // An invalid (non-multiple-of-90) runtime angle aborts at runtime instead of failing to compile.
      auto c90I64ForCheck = rewriter.create<mlir::arith::ConstantIntOp>(loc, 90, 64);
      auto c360I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 360, 64);
      auto c0I64 = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 64);

      mlir::Value remainder90 = rewriter.create<mlir::arith::RemSIOp>(loc, angle, c90I64ForCheck);
      mlir::Value isInvalidAngle =
          rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, remainder90, c0I64);
      auto ifInvalidAngle = rewriter.create<mlir::scf::IfOp>(loc, isInvalidAngle, /*withElseRegion=*/false);
      rewriter.setInsertionPointToStart(ifInvalidAngle.thenBlock());
      rewriter.create<mlir::func::CallOp>(loc, "abort", mlir::TypeRange{}, mlir::ValueRange{});
      rewriter.setInsertionPointAfter(ifInvalidAngle);

      // Normalize signed angles into [0, 360), e.g. -90 -> 270.
      mlir::Value remainder360 = rewriter.create<mlir::arith::RemSIOp>(loc, angle, c360I64);
      mlir::Value shifted = rewriter.create<mlir::arith::AddIOp>(loc, remainder360, c360I64);
      normalizedAngle = rewriter.create<mlir::arith::RemSIOp>(loc, shifted, c360I64);
    }

    mlir::Value is90 =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, normalizedAngle, c90I64);
    mlir::Value is180 =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, normalizedAngle, c180I64);
    mlir::Value is270 =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, normalizedAngle, c270I64);
    mlir::Value is90Or270 = rewriter.create<mlir::arith::OrIOp>(loc, is90, is270);

    mlir::Value outputHeight = rewriter.create<mlir::arith::SelectOp>(loc, is90Or270, inputWidth, inputHeight);
    mlir::Value outputWidth = rewriter.create<mlir::arith::SelectOp>(loc, is90Or270, inputHeight, inputWidth);

    auto kDynamic = mlir::ShapedType::kDynamic;
    auto output = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic, 4}, i8Type),
                                                         mlir::ValueRange{outputHeight, outputWidth});

    auto pixelLoop = createAffineParallel(rewriter, loc, {outputHeight, outputWidth});
    rewriter.setInsertionPointToStart(pixelLoop.getBody());

    mlir::Value pixelRowIndex = pixelLoop.getIVs()[0];
    mlir::Value pixelColIndex = pixelLoop.getIVs()[1];

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

    auto copyChannel = [&](Channel ch) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int>(ch));

      auto srcVal = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{srcRow, srcCol, cOffset});

      rewriter.create<mlir::memref::StoreOp>(loc, srcVal, output,
                                             mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});
    };

    copyChannel(Channel::R);
    copyChannel(Channel::G);
    copyChannel(Channel::B);
    copyChannel(Channel::A);

    rewriter.setInsertionPointAfter(pixelLoop);
    rewriter.replaceOp(op, output);
    return mlir::success();
  }
};

/**
 * @brief Relocates `value`'s defining chain to `funcOp`'s top level, since MLIR requires an
 * affine.parallel/affine.apply dim or symbol to be defined there. Only safe for a pure `arith`
 * chain (a compile-time-constant kernel size); fails otherwise.
 */
mlir::FailureOr<mlir::Value> hoistArithChainToFunctionTop(mlir::ConversionPatternRewriter &rewriter,
                                                          mlir::func::FuncOp funcOp, mlir::Operation *insertBefore,
                                                          mlir::Value value) {
  mlir::Region &topLevelRegion = funcOp.getBody();

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (blockArg && blockArg.getOwner()->getParent() == &topLevelRegion)
      return value;
    return mlir::failure();
  }
  if (defOp->getParentRegion() == &topLevelRegion)
    return value;

  if (!defOp->getDialect() || !mlir::isa<mlir::arith::ArithDialect>(defOp->getDialect()))
    return mlir::failure();

  mlir::IRMapping mapping;
  for (mlir::Value operand : defOp->getOperands()) {
    auto hoistedOperand = hoistArithChainToFunctionTop(rewriter, funcOp, insertBefore, operand);
    if (mlir::failed(hoistedOperand))
      return mlir::failure();
    mapping.map(operand, *hoistedOperand);
  }

  // Fixed anchor (not setInsertionPointToStart) keeps leaves ordered before their uses.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertBefore);
  mlir::Operation *cloned = rewriter.clone(*defOp, mapping);
  return cloned->getResult(mlir::cast<mlir::OpResult>(value).getResultNumber());
}

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

    mlir::Value input = operands[0];
    if (!mlir::isa<mlir::MemRefType>(input.getType())) {
      rawOp->emitOpError("expected input image to be a MemRefType");
      return mlir::failure();
    }

    mlir::Value kernelOperand = (operands.size() > 1) ? operands[1] : nullptr;

    auto neighborhoodSizeResult = op.getNeighborhoodSize(rewriter, loc, operands);
    if (!neighborhoodSizeResult) {
      rawOp->emitOpError("unable to determine a valid neighborhood size");
      return mlir::failure();
    }

    auto [neighborhoodRows, neighborhoodCols] = *neighborhoodSizeResult;

    auto indexType = rewriter.getIndexType();
    auto i8Type = rewriter.getI8Type();
    auto f64Type = rewriter.getF64Type();

    mlir::Value inputHeight = rewriter.create<mlir::memref::DimOp>(loc, input, 0);
    mlir::Value inputWidth = rewriter.create<mlir::memref::DimOp>(loc, input, 1);

    auto kDynamic = mlir::ShapedType::kDynamic;
    auto output = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic, 4}, i8Type),
                                                         mlir::ValueRange{inputHeight, inputWidth});

    mlir::Value c2 = createIntConstant(rewriter, loc, 2);
    mlir::Value neighborhoodRowRadius = rewriter.create<mlir::arith::DivSIOp>(loc, neighborhoodRows, c2);
    mlir::Value neighborhoodColRadius = rewriter.create<mlir::arith::DivSIOp>(loc, neighborhoodCols, c2);

    mlir::Value neighborhoodRowsIdx = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, neighborhoodRows);
    mlir::Value neighborhoodColsIdx = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, neighborhoodCols);
    mlir::Value neighborhoodRowRadiusIdx =
        rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, neighborhoodRowRadius);
    mlir::Value neighborhoodColRadiusIdx =
        rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, neighborhoodColRadius);

    auto funcOp = rawOp->getParentOfType<mlir::func::FuncOp>();
    mlir::Operation *hoistAnchor = &funcOp.getBody().front().front();
    for (mlir::Value *dimValue :
        {&neighborhoodRowsIdx, &neighborhoodColsIdx, &neighborhoodRowRadiusIdx, &neighborhoodColRadiusIdx}) {
      auto hoisted = hoistArithChainToFunctionTop(rewriter, funcOp, hoistAnchor, *dimValue);
      if (mlir::failed(hoisted)) {
        rawOp->emitOpError("neighborhood size must be a compile-time constant computable at the function's top "
                           "level; it cannot depend on a runtime value while nested inside a for/if block");
        return mlir::failure();
      }
      *dimValue = *hoisted;
    }

    mlir::Value zeroIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

    auto pixelLoop = createAffineParallel(rewriter, loc, {inputHeight, inputWidth});
    rewriter.setInsertionPointToStart(pixelLoop.getBody());

    mlir::Value pixelRowIndex = pixelLoop.getIVs()[0];
    mlir::Value pixelColIndex = pixelLoop.getIVs()[1];

    // The seed each kernel tap's contribution is folded into before the
    // affine.parallel reduce combines all taps; it must be the identity
    // element of getReductionKind() (e.g. 0.0 for addf, +inf for minimumf).
    mlir::Value identity = op.initializeAccumulator(rewriter, loc);
    mlir::arith::AtomicRMWKind reductionKind = op.getReductionKind();

    auto coordMap = mlir::AffineMap::get(
        2, 1, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1) - rewriter.getAffineSymbolExpr(0));

    auto kernelLoop =
        createAffineParallel(rewriter, loc, {neighborhoodRowsIdx, neighborhoodColsIdx},
                             mlir::TypeRange{f64Type, f64Type, f64Type}, {reductionKind, reductionKind, reductionKind});
    rewriter.setInsertionPointToStart(kernelLoop.getBody());

    mlir::Value kRowIndex = kernelLoop.getIVs()[0];
    mlir::Value kColIndex = kernelLoop.getIVs()[1];

    mlir::Value sampleRow = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, coordMap, mlir::ValueRange{pixelRowIndex, kRowIndex, neighborhoodRowRadiusIdx});
    mlir::Value sampleCol = rewriter.create<mlir::affine::AffineApplyOp>(
        loc, coordMap, mlir::ValueRange{pixelColIndex, kColIndex, neighborhoodColRadiusIdx});

    auto rowLow = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, sampleRow, zeroIndex);
    auto rowHigh = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, sampleRow, inputHeight);
    auto colLow = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, sampleCol, zeroIndex);
    auto colHigh = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, sampleCol, inputWidth);

    auto rowValid = rewriter.create<mlir::arith::AndIOp>(loc, rowLow, rowHigh);
    auto colValid = rewriter.create<mlir::arith::AndIOp>(loc, colLow, colHigh);
    auto isValid = rewriter.create<mlir::arith::AndIOp>(loc, rowValid, colValid);

    // Both branches must yield a per-channel contribution: the real sample
    // when in-bounds, or the reduction identity (a no-op for the reduce)
    // when the tap falls outside the image.
    auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f64Type, f64Type, f64Type}, isValid.getResult(),
                                                 /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    mlir::Value kernelWeight = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(1.0));
    if (kernelOperand && mlir::isa<mlir::MemRefType>(kernelOperand.getType())) {
      kernelWeight = rewriter.create<mlir::memref::LoadOp>(loc, kernelOperand, mlir::ValueRange{kRowIndex, kColIndex});
    }

    auto sampleChannel = [&](Channel ch) -> mlir::Value {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int>(ch));
      auto pixelByte =
          rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{sampleRow, sampleCol, cOffset});
      auto pixelAsF64 = rewriter.create<mlir::arith::UIToFPOp>(loc, f64Type, pixelByte);
      return op.accumulate(rewriter, loc, identity, pixelAsF64, kernelWeight);
    };

    mlir::Value contribR = sampleChannel(Channel::R);
    mlir::Value contribG = sampleChannel(Channel::G);
    mlir::Value contribB = sampleChannel(Channel::B);
    rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{contribR, contribG, contribB});

    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{identity, identity, identity});

    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<mlir::affine::AffineYieldOp>(loc, ifOp.getResults());

    rewriter.setInsertionPointAfter(kernelLoop);

    auto finalizeChannel = [&](Channel ch, mlir::Value channelSum) {
      auto finalizedAcc = op.finalizeAccumulator(rewriter, loc, channelSum);

      auto c0F64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(0.0));
      auto c255F64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, llvm::APFloat(255.0));
      auto clampedLow = rewriter.create<mlir::arith::MaximumFOp>(loc, finalizedAcc, c0F64);
      auto clampedHigh = rewriter.create<mlir::arith::MinimumFOp>(loc, clampedLow, c255F64);
      auto byteVal = rewriter.create<mlir::arith::FPToUIOp>(loc, i8Type, clampedHigh);

      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int>(ch));
      rewriter.create<mlir::memref::StoreOp>(loc, byteVal, output,
                                             mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});
    };

    finalizeChannel(Channel::R, kernelLoop.getResult(0));
    finalizeChannel(Channel::G, kernelLoop.getResult(1));
    finalizeChannel(Channel::B, kernelLoop.getResult(2));

    auto c3 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 3);
    auto alphaVal =
        rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{pixelRowIndex, pixelColIndex, c3});
    rewriter.create<mlir::memref::StoreOp>(loc, alphaVal, output, mlir::ValueRange{pixelRowIndex, pixelColIndex, c3});

    rewriter.setInsertionPointAfter(pixelLoop);
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

    if (operands.size() < 2) {
      rawOp->emitOpError("expected at least two operands");
      return mlir::failure();
    }

    auto lhsImage = operands[0];
    auto rhsImage = operands[1];

    if (!mlir::isa<mlir::MemRefType>(lhsImage.getType()) || !mlir::isa<mlir::MemRefType>(rhsImage.getType())) {
      rawOp->emitOpError("expected input images to be MemRefTypes");
      return mlir::failure();
    }

    auto i8Type = rewriter.getI8Type();
    auto indexType = rewriter.getIndexType();

    mlir::Value lhsHeight = rewriter.create<mlir::memref::DimOp>(loc, lhsImage, 0);
    mlir::Value lhsWidth = rewriter.create<mlir::memref::DimOp>(loc, lhsImage, 1);

    mlir::Value rhsHeight = rewriter.create<mlir::memref::DimOp>(loc, rhsImage, 0);
    mlir::Value rhsWidth = rewriter.create<mlir::memref::DimOp>(loc, rhsImage, 1);

    // Compare dimensions and abort if they don't match
    auto widthMismatch = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhsWidth, rhsWidth);
    auto heightMismatch =
        rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhsHeight, rhsHeight);
    auto dimMismatch = rewriter.create<mlir::arith::OrIOp>(loc, widthMismatch, heightMismatch);
    auto ifDimMismatch = rewriter.create<mlir::scf::IfOp>(loc, dimMismatch.getResult(), false);
    rewriter.setInsertionPointToStart(ifDimMismatch.thenBlock());
    rewriter.create<mlir::func::CallOp>(loc, "abort", mlir::TypeRange{}, mlir::ValueRange{});

    rewriter.setInsertionPointAfter(ifDimMismatch);

    if (auto blendOp = mlir::dyn_cast<BlendOp>(rawOp)) {
      mlir::Value weight = blendOp.getWeight();
      if (!weight.getDefiningOp<mlir::arith::ConstantFloatOp>()) {
        mlir::Value tooLow = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, weight,
                                                                   createFloatConstant(rewriter, loc, 0.0));
        mlir::Value tooHigh = rewriter.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, weight,
                                                                    createFloatConstant(rewriter, loc, 1.0));
        mlir::Value outOfRange = rewriter.create<mlir::arith::OrIOp>(loc, tooLow, tooHigh);
        auto ifOutOfRange = rewriter.create<mlir::scf::IfOp>(loc, outOfRange, /*withElseRegion=*/false);
        rewriter.setInsertionPointToStart(ifOutOfRange.thenBlock());
        rewriter.create<mlir::func::CallOp>(loc, "abort", mlir::TypeRange{}, mlir::ValueRange{});
        rewriter.setInsertionPointAfter(ifOutOfRange);
      }
    }

    auto kDynamic = mlir::ShapedType::kDynamic;
    auto output = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic, 4}, i8Type),
                                                         mlir::ValueRange{lhsHeight, lhsWidth});

    auto pixelLoop = createAffineParallel(rewriter, loc, {lhsHeight, lhsWidth});
    rewriter.setInsertionPointToStart(pixelLoop.getBody());

    mlir::Value pixelRowIndex = pixelLoop.getIVs()[0];
    mlir::Value pixelColIndex = pixelLoop.getIVs()[1];

    auto processChannel = [&](Channel ch) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int>(ch));

      auto lhsByte =
          rewriter.create<mlir::memref::LoadOp>(loc, lhsImage, mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});
      auto rhsByte =
          rewriter.create<mlir::memref::LoadOp>(loc, rhsImage, mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});

      auto resultByte = op.transformPixels(rewriter, loc, lhsByte, rhsByte, ch);

      rewriter.create<mlir::memref::StoreOp>(loc, resultByte, output,
                                             mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});
    };

    processChannel(Channel::R);
    processChannel(Channel::G);
    processChannel(Channel::B);
    processChannel(Channel::A);

    rewriter.setInsertionPointAfter(pixelLoop);
    rewriter.replaceOp(rawOp, output);

    return mlir::success();
  }
};

struct ElementWiseUnaryOpToAffine : mlir::OpInterfaceConversionPattern<ElementWiseUnaryOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  mlir::LogicalResult matchAndRewrite(ElementWiseUnaryOpInterface op, mlir::ArrayRef<mlir::Value> operands,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto *rawOp = op.getOperation();
    mlir::Location loc = rawOp->getLoc();

    if (operands.empty()) {
      rawOp->emitOpError("expected at least one operand");
      return mlir::failure();
    }

    auto input = operands[0];
    if (!mlir::isa<mlir::MemRefType>(input.getType())) {
      rawOp->emitOpError("expected input image to be a MemRefType");
      return mlir::failure();
    }

    auto i8Type = rewriter.getI8Type();

    mlir::Value inputHeight = rewriter.create<mlir::memref::DimOp>(loc, input, 0);
    mlir::Value inputWidth = rewriter.create<mlir::memref::DimOp>(loc, input, 1);

    auto kDynamic = mlir::ShapedType::kDynamic;
    auto output = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic, 4}, i8Type),
                                                         mlir::ValueRange{inputHeight, inputWidth});

    auto pixelLoop = createAffineParallel(rewriter, loc, {inputHeight, inputWidth});
    rewriter.setInsertionPointToStart(pixelLoop.getBody());

    mlir::Value pixelRowIndex = pixelLoop.getIVs()[0];
    mlir::Value pixelColIndex = pixelLoop.getIVs()[1];

    auto processChannel = [&](Channel ch) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int>(ch));

      auto inputByte =
          rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});

      mlir::Value finalValue;
      if (ch != Channel::A) {
        // Convert to float64 for the interface's transformPixel computation, then cast back to i8
        auto inputF64 = rewriter.create<mlir::arith::UIToFPOp>(loc, rewriter.getF64Type(), inputByte);
        mlir::Value transformedF64 = op.transformPixel(rewriter, loc, inputF64);

        auto c0F64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, rewriter.getF64Type(), llvm::APFloat(0.0));
        auto c255F64 = rewriter.create<mlir::arith::ConstantFloatOp>(loc, rewriter.getF64Type(), llvm::APFloat(255.0));
        auto clampedLow = rewriter.create<mlir::arith::MaximumFOp>(loc, transformedF64, c0F64);
        auto clampedHigh = rewriter.create<mlir::arith::MinimumFOp>(loc, clampedLow, c255F64);
        finalValue = rewriter.create<mlir::arith::FPToUIOp>(loc, i8Type, clampedHigh);
      } else {
        finalValue = inputByte;
      }

      rewriter.create<mlir::memref::StoreOp>(loc, finalValue, output,
                                             mlir::ValueRange{pixelRowIndex, pixelColIndex, cOffset});
    };

    processChannel(Channel::R);
    processChannel(Channel::G);
    processChannel(Channel::B);
    processChannel(Channel::A);

    rewriter.setInsertionPointAfter(pixelLoop);
    rewriter.replaceOp(rawOp, output);

    return mlir::success();
  }
};

struct CropToAffine : mlir::OpConversionPattern<CropOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(CropOp op, CropOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto i8Type = rewriter.getI8Type();
    auto indexType = rewriter.getIndexType();

    mlir::Value input = adaptor.getInput();
    if (!mlir::isa<mlir::MemRefType>(input.getType())) {
      op.emitOpError("expected input image to be a MemRefType");
      return mlir::failure();
    }

    mlir::Value inputHeight = rewriter.create<mlir::memref::DimOp>(loc, input, 0);
    mlir::Value inputWidth = rewriter.create<mlir::memref::DimOp>(loc, input, 1);
    (void)inputWidth;
    (void)inputHeight;

    mlir::Value xI32 = adaptor.getX();
    mlir::Value yI32 = adaptor.getY();
    mlir::Value cropWI32 = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI32Type(), adaptor.getWidth());
    mlir::Value cropHI32 = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI32Type(), adaptor.getHeight());

    mlir::Value xIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, xI32);
    mlir::Value yIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, yI32);
    mlir::Value cropW = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, cropWI32);
    mlir::Value cropH = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, cropHI32);

    auto kDynamic = mlir::ShapedType::kDynamic;
    auto output = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic, 4}, i8Type),
                                                         mlir::ValueRange{cropH, cropW});

    auto pixelLoop = createAffineParallel(rewriter, loc, {cropH, cropW});
    rewriter.setInsertionPointToStart(pixelLoop.getBody());

    mlir::Value outRow = pixelLoop.getIVs()[0];
    mlir::Value outCol = pixelLoop.getIVs()[1];

    // Map output pixel (outRow, outCol) to input pixel (srcRow, srcCol)
    mlir::Value srcRow = rewriter.create<mlir::arith::AddIOp>(loc, yIndex, outRow);
    mlir::Value srcCol = rewriter.create<mlir::arith::AddIOp>(loc, xIndex, outCol);

    auto copyChannel = [&](Channel ch) {
      auto cOffset = rewriter.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int>(ch));

      auto srcVal = rewriter.create<mlir::memref::LoadOp>(loc, input, mlir::ValueRange{srcRow, srcCol, cOffset});

      rewriter.create<mlir::memref::StoreOp>(loc, srcVal, output, mlir::ValueRange{outRow, outCol, cOffset});
    };

    copyChannel(Channel::R);
    copyChannel(Channel::G);
    copyChannel(Channel::B);
    copyChannel(Channel::A);

    rewriter.setInsertionPointAfter(pixelLoop);
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
                           mlir::func::FuncDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                           mlir::math::MathDialect>();

    target.addLegalOp<mlir::UnrealizedConversionCastOp, StringConstOp>();
    target.addIllegalDialect<PiccelerDialect>();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<ElementWiseUnaryOpToAffine>(typeConverter, ctx);
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
