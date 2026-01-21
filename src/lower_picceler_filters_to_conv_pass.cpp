#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

#include "ops.h"
#include "types.h"

namespace picceler {

struct SharpenToConvolution : mlir::OpConversionPattern<SharpenOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(SharpenOp op, SharpenOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value input = adaptor.getInput();
    mlir::Value strengthValue = adaptor.getValue();
    auto f64Type = rewriter.getF64Type();

    auto constOp = strengthValue.getDefiningOp<mlir::arith::ConstantIntOp>();
    
    if (!constOp) {
      return op.emitError("Sharpen supports only constant integer strength values for now.");
    }

    int64_t strength = constOp.value();
    double a = static_cast<double>(strength) / 25.0;
    
    double center = 1.0 + 4.0 * a;
    double neighbor = -a;

    double kernelValues[] = {
         0.0,    neighbor, 0.0,
        neighbor, center,  neighbor,
         0.0,    neighbor, 0.0
    };

    auto tensorType = mlir::RankedTensorType::get({3, 3}, f64Type);
    auto dataAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(kernelValues));

    auto kernelType = picceler::KernelType::get(getContext(), 3, 3);
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
  patterns.add<SharpenToConvolution>(ctx);

  mlir::ConversionTarget target(*ctx);
  target.addIllegalOp<SharpenOp>();
  target.addLegalOp<ConvolutionOp>();
  target.addLegalOp<KernelConstOp>();

  if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace picceler
