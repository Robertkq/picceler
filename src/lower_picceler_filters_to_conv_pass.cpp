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

using Kernel3x3 = std::array<double, 9>;

mlir::FailureOr<Kernel3x3> calculateSharpenKernel(SharpenOp op, SharpenOpAdaptor adaptor) {
    mlir::Value strengthValue = adaptor.getValue();

    auto constOp = strengthValue.getDefiningOp<mlir::arith::ConstantIntOp>();
    if (!constOp) {
        op.emitError("Sharpen supports only constant integer strength values.");
        return mlir::failure();
    }

    double strength = static_cast<double>(constOp.value()) / 25.0;
    double center = 1.0 + 4.0 * strength;
    double neighbor = -strength;

    return Kernel3x3{
         0.0,    neighbor, 0.0,
        neighbor, center,  neighbor,
         0.0,    neighbor, 0.0
    };
}

template <typename OpTy>
struct FilterToConvolutionPattern : mlir::OpConversionPattern<OpTy> {
  using KernelCalculator = std::function<mlir::FailureOr<Kernel3x3>(OpTy, typename OpTy::Adaptor)>;

  KernelCalculator kernelCalc;

  FilterToConvolutionPattern(mlir::MLIRContext *ctx, KernelCalculator calc)
      : mlir::OpConversionPattern<OpTy>(ctx), kernelCalc(calc) {}

  mlir::LogicalResult matchAndRewrite(OpTy op, OpTy::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = adaptor.getInput();
    mlir::Value strengthValue = adaptor.getValue();
    auto f64Type = rewriter.getF64Type();

    auto kernelRes = kernelCalc(op, adaptor);
    if (mlir::failed(kernelRes)) {
      return mlir::failure();
    }
    Kernel3x3 kernelValues = *kernelRes;

    auto tensorType = mlir::RankedTensorType::get({3, 3}, f64Type);
    auto dataAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(kernelValues));

    auto kernelType = picceler::KernelType::get(op.getContext(), 3, 3);
    auto kernelOp = rewriter.create<picceler::KernelConstOp>(loc, kernelType, dataAttr);

    rewriter.replaceOpWithNewOp<ConvolutionOp>(
        op,
        op.getType(),
        input,
        kernelOp.getResult()
    );

    return mlir::success();
  }
};

void LowerPiccelerFiltersToConvPass::runOnOperation() {
  spdlog::trace("Running LowerPiccelerFiltersToConvPass");
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = &getContext();

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<FilterToConvolutionPattern<SharpenOp>>(ctx, calculateSharpenKernel);

  mlir::ConversionTarget target(*ctx);
  target.addIllegalOp<SharpenOp>();
  target.addLegalOp<ConvolutionOp>();
  target.addLegalOp<KernelConstOp>();

  if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace picceler
