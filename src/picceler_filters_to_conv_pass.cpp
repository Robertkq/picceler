#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

#include "ops.h"
#include "types.h"

namespace picceler {

struct KernelData {
  int64_t rows;
  int64_t cols;
  std::vector<double> values;
};

mlir::FailureOr<KernelData> calculateSharpenKernel(SharpenOp op, SharpenOpAdaptor adaptor) {
  mlir::Value strengthValue = adaptor.getValue();

  auto constOp = strengthValue.getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constOp) {
    op.emitError("Sharpen supports only constant integer strength values.");
    return mlir::failure();
  }

  double strength = static_cast<double>(constOp.value()) / 25.0;
  double center = 1.0 + 4.0 * strength;
  double neighbor = -strength;

  return KernelData{3, 3, {0.0, neighbor, 0.0, neighbor, center, neighbor, 0.0, neighbor, 0.0}};
}

mlir::FailureOr<KernelData> calculateBoxBlurKernel(BoxBlurOp op, BoxBlurOpAdaptor adaptor) {
  int radius = 1;

  auto constOp = adaptor.getRadius().getDefiningOp<mlir::arith::ConstantIntOp>();
  if (constOp) {
    radius = constOp.value();
  }

  if (radius < 1)
    radius = 1;

  if (radius > 500) {
    return op.emitError("Box blur radius is too large (" + std::to_string(radius) + "). Maximum allowed is 500.");
  }

  int64_t size = 2 * radius + 1;
  double val = 1.0 / static_cast<double>(size * size);

  std::vector<double> values(size * size, val);

  return KernelData{size, size, values};
}

mlir::FailureOr<KernelData> calculateGaussianKernel(GaussianBlurOp op, GaussianBlurOpAdaptor adaptor) {
  int radius = 2;

  auto constOp = adaptor.getRadius().getDefiningOp<mlir::arith::ConstantIntOp>();
  if (constOp)
    radius = constOp.value();

  if (radius < 1)
    radius = 1;

  if (radius > 500) {
    return op.emitError("Gaussian blur radius is too large (" + std::to_string(radius) + "). Maximum allowed is 500.");
  }

  int64_t size = 2 * radius + 1;
  double sigma = static_cast<double>(radius) / 2.0;
  if (sigma < 0.5)
    sigma = 0.5;

  std::vector<double> values;
  values.reserve(size * size);

  double sum = 0.0;

  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      double exponent = -(x * x + y * y) / (2 * sigma * sigma);
      double value = std::exp(exponent) / (2 * M_PI * sigma * sigma);
      values.push_back(value);
      sum += value;
    }
  }

  for (double &v : values) {
    v /= sum;
  }

  return KernelData{size, size, values};
}

mlir::FailureOr<KernelData> calculateEdgeDetectKernel(EdgeDetectOp op, EdgeDetectOpAdaptor adaptor) {
  return KernelData{3, 3, {-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0}};
}

mlir::FailureOr<KernelData> calculateEmbossKernel(EmbossOp op, EmbossOpAdaptor adaptor) {
  return KernelData{3, 3, {-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0}};
}

template <typename OpTy> struct FilterToConvolutionPattern : mlir::OpConversionPattern<OpTy> {
  using KernelCalculator = std::function<mlir::FailureOr<KernelData>(OpTy, typename OpTy::Adaptor)>;

  KernelCalculator kernelCalc;

  FilterToConvolutionPattern(mlir::MLIRContext *ctx, KernelCalculator calc)
      : mlir::OpConversionPattern<OpTy>(ctx), kernelCalc(calc) {}

  mlir::LogicalResult matchAndRewrite(OpTy op, OpTy::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = adaptor.getInput();
    auto f64Type = rewriter.getF64Type();

    auto kernelRes = kernelCalc(op, adaptor);
    if (mlir::failed(kernelRes)) {
      return mlir::failure();
    }

    KernelData kData = *kernelRes;
    if (kData.values.size() != kData.rows * kData.cols) {
      return op.emitError("Kernel dimensions do not match the number of elements.");
    }

    auto tensorType = mlir::RankedTensorType::get({kData.rows, kData.cols}, f64Type);
    auto dataAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(kData.values));

    auto kernelType = picceler::KernelType::get(op.getContext(), kData.rows, kData.cols);
    auto kernelOp = rewriter.create<picceler::KernelConstOp>(loc, kernelType, dataAttr);

    rewriter.replaceOpWithNewOp<ConvolutionOp>(op, op.getType(), input, kernelOp.getResult());

    return mlir::success();
  }
};

#define GEN_PASS_DEF_PICCELERFILTERSTOCONV
#include "piccelerPasses.h.inc"

struct PiccelerFiltersToConvPass : public impl::PiccelerFiltersToConvBase<PiccelerFiltersToConvPass> {

  void runOnOperation() override {
    spdlog::trace("Running PiccelerFiltersToConvPass");
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<FilterToConvolutionPattern<SharpenOp>>(ctx, calculateSharpenKernel);
    patterns.add<FilterToConvolutionPattern<BoxBlurOp>>(ctx, calculateBoxBlurKernel);
    patterns.add<FilterToConvolutionPattern<GaussianBlurOp>>(ctx, calculateGaussianKernel);
    patterns.add<FilterToConvolutionPattern<EdgeDetectOp>>(ctx, calculateEdgeDetectKernel);
    patterns.add<FilterToConvolutionPattern<EmbossOp>>(ctx, calculateEmbossKernel);

    mlir::ConversionTarget target(*ctx);
    target.addIllegalOp<SharpenOp>();
    target.addIllegalOp<BoxBlurOp>();
    target.addIllegalOp<GaussianBlurOp>();
    target.addIllegalOp<EdgeDetectOp>();
    target.addIllegalOp<EmbossOp>();
    target.addLegalOp<ConvolutionOp>();
    target.addLegalOp<KernelConstOp>();

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerFiltersToConvPass() { return std::make_unique<PiccelerFiltersToConvPass>(); }
} // namespace picceler
