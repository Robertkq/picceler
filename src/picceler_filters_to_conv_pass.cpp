#include "passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

#include "spdlog/spdlog.h"

#include "ops.h"
#include "types.h"

#include <error.h>
#include <expected>
#include <format>
#include <numbers>

namespace picceler {

class KernelData {
public:
  KernelData(int64_t rows, int64_t cols, std::vector<double> values)
      : _rows(rows), _cols(cols), _values(std::move(values)) {}

  int64_t rows() const { return _rows; }
  int64_t cols() const { return _cols; }
  const std::vector<double> &values() const { return _values; }

private:
  int64_t _rows;
  int64_t _cols;
  std::vector<double> _values;
};

Result<KernelData> calculateSharpenKernel(SharpenOp op, SharpenOpAdaptor adaptor) {
  mlir::Value strengthValue = adaptor.getValue();

  auto constOp = strengthValue.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constOp) {
    return std::unexpected(CompileError("Sharpen supports only constant integer strength values."));
  }

  double strength = static_cast<double>(constOp.value()) / 25.0;
  double center = 1.0 + 4.0 * strength;
  double neighbor = -strength;

  return KernelData{3, 3, {0.0, neighbor, 0.0, neighbor, center, neighbor, 0.0, neighbor, 0.0}};
}

Result<KernelData> calculateBoxBlurKernel(BoxBlurOp op, BoxBlurOpAdaptor adaptor) {

  auto constOp = adaptor.getRadius().getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constOp) {
    return std::unexpected(CompileError("Box blur supports only constant integer radius values."));
  }
  int64_t radius = constOp.value();

  if (radius < 1) {

    return std::unexpected(
        CompileError(std::format("Box blur radius must be at least 1. Given: {}", std::to_string(radius))));
  }

  if (radius > 500) {
    return std::unexpected(CompileError(
        std::format("Box blur radius is too large ({}). Maximum allowed is 500.", std::to_string(radius))));
  }

  int64_t size = 2 * radius + 1;
  double val = 1.0 / static_cast<double>(size * size);

  std::vector<double> values(size * size, val);

  return KernelData{size, size, std::move(values)};
}

Result<KernelData> calculateGaussianKernel(GaussianBlurOp op, GaussianBlurOpAdaptor adaptor) {

  auto constOp = adaptor.getRadius().getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constOp) {
    return std::unexpected(CompileError("Gaussian blur supports only constant integer radius values."));
  }
  int64_t radius = constOp.value();

  if (radius < 1) {
    return std::unexpected(
        CompileError(std::format("Gaussian blur radius must be at least 1. Given: {}", std::to_string(radius))));
  }

  if (radius > 500) {
    return std::unexpected(CompileError(
        std::format("Gaussian blur radius is too large ({}). Maximum allowed is 500.", std::to_string(radius))));
  }

  int64_t size = 2 * radius + 1;
  double sigma = static_cast<double>(radius) / 2.0;
  if (sigma < 0.5)
    sigma = 0.5;

  std::vector<double> values;
  values.reserve(size * size);

  double sum = 0.0;

  for (int64_t y = -radius; y <= radius; ++y) {
    for (int64_t x = -radius; x <= radius; ++x) {
      double exponent = static_cast<double>(-(x * x + y * y)) / (2 * sigma * sigma);
      double value = std::exp(exponent) / (2 * std::numbers::pi * sigma * sigma);
      values.push_back(value);
      sum += value;
    }
  }

  for (double &v : values) {
    v /= sum;
  }

  return KernelData{size, size, std::move(values)};
}

Result<KernelData> calculateEdgeDetectKernel(EdgeDetectOp op, EdgeDetectOpAdaptor adaptor) {
  return KernelData{3, 3, {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0}};
}

Result<KernelData> calculateEmbossKernel(EmbossOp op, EmbossOpAdaptor adaptor) {
  return KernelData{3, 3, {-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0}};
}

void emitRuntimeRadiusBoundsCheck(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, mlir::Value radius) {
  mlir::Value tooSmall = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, radius,
                                                               createIntConstant(rewriter, loc, 1));
  mlir::Value tooLarge = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, radius,
                                                               createIntConstant(rewriter, loc, 500));
  mlir::Value outOfRange = rewriter.create<mlir::arith::OrIOp>(loc, tooSmall, tooLarge);
  auto ifOutOfRange = rewriter.create<mlir::scf::IfOp>(loc, outOfRange, /*withElseRegion=*/false);
  rewriter.setInsertionPointToStart(ifOutOfRange.thenBlock());
  rewriter.create<mlir::func::CallOp>(loc, "abort", mlir::TypeRange{}, mlir::ValueRange{});
  rewriter.setInsertionPointAfter(ifOutOfRange);
}

Result<mlir::Value> buildBoxBlurKernelDynamic(BoxBlurOp op, BoxBlurOpAdaptor adaptor,
                                              mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
  mlir::Value radius = adaptor.getRadius();
  auto indexType = rewriter.getIndexType();
  auto f64Type = rewriter.getF64Type();

  emitRuntimeRadiusBoundsCheck(rewriter, loc, radius);

  mlir::Value doubleRadius = rewriter.create<mlir::arith::AddIOp>(loc, radius, radius);
  mlir::Value sizeI64 = rewriter.create<mlir::arith::AddIOp>(loc, doubleRadius, createIntConstant(rewriter, loc, 1));
  mlir::Value sizeIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, sizeI64);

  auto kDynamic = mlir::ShapedType::kDynamic;
  auto kernelMemref = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic}, f64Type),
                                                              mlir::ValueRange{sizeIndex, sizeIndex});

  mlir::Value sizeF64 = rewriter.create<mlir::arith::SIToFPOp>(loc, f64Type, sizeI64);
  mlir::Value area = rewriter.create<mlir::arith::MulFOp>(loc, sizeF64, sizeF64);
  mlir::Value uniformWeight = rewriter.create<mlir::arith::DivFOp>(loc, createFloatConstant(rewriter, loc, 1.0), area);

  auto fillLoop = createAffineParallel(rewriter, loc, {sizeIndex, sizeIndex});
  rewriter.setInsertionPointToStart(fillLoop.getBody());
  rewriter.create<mlir::memref::StoreOp>(loc, uniformWeight, kernelMemref.getResult(),
                                         mlir::ValueRange{fillLoop.getIVs()[0], fillLoop.getIVs()[1]});
  rewriter.setInsertionPointAfter(fillLoop);

  return kernelMemref.getResult();
}

Result<mlir::Value> buildGaussianKernelDynamic(GaussianBlurOp op, GaussianBlurOpAdaptor adaptor,
                                               mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
  mlir::Value radius = adaptor.getRadius();
  auto indexType = rewriter.getIndexType();
  auto i64Type = rewriter.getI64Type();
  auto f64Type = rewriter.getF64Type();

  emitRuntimeRadiusBoundsCheck(rewriter, loc, radius);

  mlir::Value doubleRadius = rewriter.create<mlir::arith::AddIOp>(loc, radius, radius);
  mlir::Value sizeI64 = rewriter.create<mlir::arith::AddIOp>(loc, doubleRadius, createIntConstant(rewriter, loc, 1));
  mlir::Value sizeIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, sizeI64);
  mlir::Value radiusIndex = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, radius);

  auto kDynamic = mlir::ShapedType::kDynamic;
  auto kernelMemref = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({kDynamic, kDynamic}, f64Type),
                                                              mlir::ValueRange{sizeIndex, sizeIndex});

  mlir::Value radiusF64 = rewriter.create<mlir::arith::SIToFPOp>(loc, f64Type, radius);
  mlir::Value halvedRadius =
      rewriter.create<mlir::arith::DivFOp>(loc, radiusF64, createFloatConstant(rewriter, loc, 2.0));
  mlir::Value sigma =
      rewriter.create<mlir::arith::MaximumFOp>(loc, halvedRadius, createFloatConstant(rewriter, loc, 0.5));
  mlir::Value sigmaSq = rewriter.create<mlir::arith::MulFOp>(loc, sigma, sigma);
  mlir::Value twoSigmaSq = rewriter.create<mlir::arith::MulFOp>(loc, createFloatConstant(rewriter, loc, 2.0), sigmaSq);
  mlir::Value normalizer = rewriter.create<mlir::arith::MulFOp>(
      loc, createFloatConstant(rewriter, loc, 2.0 * std::numbers::pi), sigmaSq);

  auto sumLoop = createAffineParallel(rewriter, loc, {sizeIndex, sizeIndex}, mlir::TypeRange{f64Type},
                                      {mlir::arith::AtomicRMWKind::addf});
  rewriter.setInsertionPointToStart(sumLoop.getBody());

  mlir::Value kRow = sumLoop.getIVs()[0];
  mlir::Value kCol = sumLoop.getIVs()[1];

  mlir::Value dy = rewriter.create<mlir::arith::SubIOp>(loc, kRow, radiusIndex);
  mlir::Value dx = rewriter.create<mlir::arith::SubIOp>(loc, kCol, radiusIndex);
  mlir::Value dyF64 = rewriter.create<mlir::arith::SIToFPOp>(
      loc, f64Type, rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, dy));
  mlir::Value dxF64 = rewriter.create<mlir::arith::SIToFPOp>(
      loc, f64Type, rewriter.create<mlir::arith::IndexCastOp>(loc, i64Type, dx));

  mlir::Value distSq = rewriter.create<mlir::arith::AddFOp>(loc, rewriter.create<mlir::arith::MulFOp>(loc, dyF64, dyF64),
                                                             rewriter.create<mlir::arith::MulFOp>(loc, dxF64, dxF64));
  mlir::Value exponent =
      rewriter.create<mlir::arith::DivFOp>(loc, rewriter.create<mlir::arith::NegFOp>(loc, distSq), twoSigmaSq);
  mlir::Value expVal = rewriter.create<mlir::math::ExpOp>(loc, exponent);
  mlir::Value rawWeight = rewriter.create<mlir::arith::DivFOp>(loc, expVal, normalizer);

  rewriter.create<mlir::memref::StoreOp>(loc, rawWeight, kernelMemref.getResult(), mlir::ValueRange{kRow, kCol});
  rewriter.create<mlir::affine::AffineYieldOp>(loc, mlir::ValueRange{rawWeight});

  rewriter.setInsertionPointAfter(sumLoop);
  mlir::Value sum = sumLoop.getResult(0);

  auto normalizeLoop = createAffineParallel(rewriter, loc, {sizeIndex, sizeIndex});
  rewriter.setInsertionPointToStart(normalizeLoop.getBody());

  mlir::Value nRow = normalizeLoop.getIVs()[0];
  mlir::Value nCol = normalizeLoop.getIVs()[1];
  mlir::Value raw = rewriter.create<mlir::memref::LoadOp>(loc, kernelMemref.getResult(), mlir::ValueRange{nRow, nCol});
  mlir::Value normalized = rewriter.create<mlir::arith::DivFOp>(loc, raw, sum);
  rewriter.create<mlir::memref::StoreOp>(loc, normalized, kernelMemref.getResult(), mlir::ValueRange{nRow, nCol});

  rewriter.setInsertionPointAfter(normalizeLoop);

  return kernelMemref.getResult();
}

Result<mlir::Value> buildSharpenKernelDynamic(SharpenOp op, SharpenOpAdaptor adaptor,
                                              mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) {
  mlir::Value strength = adaptor.getValue();
  auto f64Type = rewriter.getF64Type();

  mlir::Value strengthF64 = rewriter.create<mlir::arith::SIToFPOp>(loc, f64Type, strength);
  mlir::Value strengthNorm =
      rewriter.create<mlir::arith::DivFOp>(loc, strengthF64, createFloatConstant(rewriter, loc, 25.0));
  mlir::Value center = rewriter.create<mlir::arith::AddFOp>(
      loc, createFloatConstant(rewriter, loc, 1.0),
      rewriter.create<mlir::arith::MulFOp>(loc, createFloatConstant(rewriter, loc, 4.0), strengthNorm));
  mlir::Value neighbor = rewriter.create<mlir::arith::NegFOp>(loc, strengthNorm);
  mlir::Value zero = createFloatConstant(rewriter, loc, 0.0);

  auto kernelMemref = rewriter.create<mlir::memref::AllocaOp>(loc, mlir::MemRefType::get({3, 3}, f64Type));

  auto idx = [&](int64_t v) { return rewriter.create<mlir::arith::ConstantIndexOp>(loc, v); };
  auto store = [&](int64_t r, int64_t c, mlir::Value val) {
    rewriter.create<mlir::memref::StoreOp>(loc, val, kernelMemref.getResult(), mlir::ValueRange{idx(r), idx(c)});
  };

  store(0, 0, zero);
  store(0, 1, neighbor);
  store(0, 2, zero);
  store(1, 0, neighbor);
  store(1, 1, center);
  store(1, 2, neighbor);
  store(2, 0, zero);
  store(2, 1, neighbor);
  store(2, 2, zero);

  return kernelMemref.getResult();
}

template <typename OpTy> struct FixedKernelFilterPattern : mlir::OpConversionPattern<OpTy> {
  using KernelCalculator = std::function<Result<KernelData>(OpTy, typename OpTy::Adaptor)>;

  FixedKernelFilterPattern(mlir::MLIRContext *ctx, KernelCalculator calc)
      : mlir::OpConversionPattern<OpTy>(ctx), _kernelCalc(std::move(calc)) {}

  mlir::LogicalResult matchAndRewrite(OpTy op, OpTy::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = adaptor.getInput();
    auto f64Type = rewriter.getF64Type();

    auto kernelRes = _kernelCalc(op, adaptor);
    if (!kernelRes) {
      spdlog::error("Couldnt get kernel calculater function: {}", kernelRes.error().message());
      return mlir::failure();
    }

    // TODO: remove NOLINT comment once we return Result<T>, clang-tidy should be able to understand that we checked for
    // failure above.
    const KernelData &kData = *kernelRes; // NOLINT(bugprone-unchecked-optional-access)
    if (kData.values().size() != kData.rows() * kData.cols()) {
      op.emitError("Kernel dimensions do not match the number of elements.");
      return mlir::failure();
    }

    auto tensorType = mlir::RankedTensorType::get({kData.rows(), kData.cols()}, f64Type);
    auto dataAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(kData.values()));

    auto kernelType = picceler::KernelType::get(op.getContext(), kData.rows(), kData.cols());
    auto kernelOp = rewriter.create<picceler::KernelConstOp>(loc, kernelType, dataAttr);

    rewriter.replaceOpWithNewOp<ConvolutionOp>(op, op.getType(), input, kernelOp.getResult());

    return mlir::success();
  }

private:
  KernelCalculator _kernelCalc;
};

/**
 * @brief Rewrites a filter op whose kernel depends on a single I64 parameter (sharpen's strength,
 * box_blur/gaussian_blur's radius) into a convolution + kernel pair, using `staticCalc` when the
 * parameter is a compile-time constant and `dynamicBuilder` otherwise.
 */
template <typename OpTy> struct ParameterizedFilterPattern : mlir::OpConversionPattern<OpTy> {
  using ParamExtractor = std::function<mlir::Value(typename OpTy::Adaptor &)>;
  using StaticKernelCalculator = std::function<Result<KernelData>(OpTy, typename OpTy::Adaptor)>;
  using DynamicKernelBuilder = std::function<Result<mlir::Value>(OpTy, typename OpTy::Adaptor,
                                                                  mlir::ConversionPatternRewriter &, mlir::Location)>;

  ParameterizedFilterPattern(mlir::MLIRContext *ctx, ParamExtractor paramExtractor,
                                      StaticKernelCalculator staticCalc, DynamicKernelBuilder dynamicBuilder)
      : mlir::OpConversionPattern<OpTy>(ctx), _paramExtractor(std::move(paramExtractor)),
        _staticCalc(std::move(staticCalc)), _dynamicBuilder(std::move(dynamicBuilder)) {}

  mlir::LogicalResult matchAndRewrite(OpTy op, OpTy::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = adaptor.getInput();

    mlir::Value param = _paramExtractor(adaptor);
    mlir::APInt constantParam;
    if (mlir::matchPattern(param, mlir::m_ConstantInt(&constantParam))) {
      auto kernelRes = _staticCalc(op, adaptor);
      if (!kernelRes) {
        spdlog::error("Couldnt get kernel calculater function: {}", kernelRes.error().message());
        return mlir::failure();
      }

      // TODO: remove NOLINT comment once we return Result<T>, clang-tidy should be able to understand that we
      // checked for failure above.
      const KernelData &kData = *kernelRes; // NOLINT(bugprone-unchecked-optional-access)
      if (kData.values().size() != kData.rows() * kData.cols()) {
        op.emitError("Kernel dimensions do not match the number of elements.");
        return mlir::failure();
      }

      auto f64Type = rewriter.getF64Type();
      auto tensorType = mlir::RankedTensorType::get({kData.rows(), kData.cols()}, f64Type);
      auto dataAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(kData.values()));

      auto kernelType = picceler::KernelType::get(op.getContext(), kData.rows(), kData.cols());
      auto kernelOp = rewriter.create<picceler::KernelConstOp>(loc, kernelType, dataAttr);

      rewriter.replaceOpWithNewOp<ConvolutionOp>(op, op.getType(), input, kernelOp.getResult());
      return mlir::success();
    }

    auto kernelValueRes = _dynamicBuilder(op, adaptor, rewriter, loc);
    if (!kernelValueRes) {
      spdlog::error("Couldnt build a runtime kernel: {}", kernelValueRes.error().message());
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ConvolutionOp>(op, op.getType(), input, *kernelValueRes);
    return mlir::success();
  }

private:
  ParamExtractor _paramExtractor;
  StaticKernelCalculator _staticCalc;
  DynamicKernelBuilder _dynamicBuilder;
};

#define GEN_PASS_DEF_PICCELERFILTERSTOCONV
#include "piccelerPasses.h.inc"

/**
 * @brief A pass that converts high-level image filter operations (like sharpen, blur, edge detect) into convolution
 * operations with constant kernels.
 */
struct PiccelerFiltersToConvPass : public impl::PiccelerFiltersToConvBase<PiccelerFiltersToConvPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    if (!module.lookupSymbol<mlir::func::FuncOp>("abort")) {
      mlir::OpBuilder builder(&module.getBodyRegion());
      auto func = builder.create<mlir::func::FuncOp>(module.getLoc(), "abort", builder.getFunctionType({}, {}));
      func.setPrivate();
    }

    mlir::RewritePatternSet patterns(ctx);

    patterns.add<ParameterizedFilterPattern<SharpenOp>>(
        ctx, [](SharpenOpAdaptor &adaptor) { return adaptor.getValue(); }, calculateSharpenKernel,
        buildSharpenKernelDynamic);
    patterns.add<ParameterizedFilterPattern<BoxBlurOp>>(
        ctx, [](BoxBlurOpAdaptor &adaptor) { return adaptor.getRadius(); }, calculateBoxBlurKernel,
        buildBoxBlurKernelDynamic);
    patterns.add<ParameterizedFilterPattern<GaussianBlurOp>>(
        ctx, [](GaussianBlurOpAdaptor &adaptor) { return adaptor.getRadius(); }, calculateGaussianKernel,
        buildGaussianKernelDynamic);

    patterns.add<FixedKernelFilterPattern<EdgeDetectOp>>(ctx, calculateEdgeDetectKernel);
    patterns.add<FixedKernelFilterPattern<EmbossOp>>(ctx, calculateEmbossKernel);

    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<SharpenOp>();
    target.addIllegalOp<BoxBlurOp>();
    target.addIllegalOp<GaussianBlurOp>();
    target.addIllegalOp<EdgeDetectOp>();
    target.addIllegalOp<EmbossOp>();
    target.addLegalOp<ConvolutionOp>();
    target.addLegalOp<KernelConstOp>();
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::memref::MemRefDialect, mlir::affine::AffineDialect,
                           mlir::math::MathDialect, mlir::scf::SCFDialect, mlir::func::FuncDialect>();

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerFiltersToConvPass() { return std::make_unique<PiccelerFiltersToConvPass>(); }
} // namespace picceler
