#include "passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ops.h"
#include "types.h"

namespace picceler {

struct KernelToMemref : public mlir::OpConversionPattern<KernelConstOp> {
  using OpConversionPattern<KernelConstOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(KernelConstOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto f64Type = rewriter.getF64Type();

    auto kernelType = mlir::dyn_cast<KernelType>(op.getResult().getType());
    if (!kernelType)
      return mlir::failure();

    int64_t cols = kernelType.getCols();

    // Safe conversion check
    auto convertedType = getTypeConverter()->convertType(kernelType);
    if (!convertedType)
      return mlir::failure();

    auto memrefType = mlir::cast<mlir::MemRefType>(convertedType);
    auto allocaOp = rewriter.create<mlir::memref::AllocaOp>(loc, memrefType);
    mlir::Value kStack = allocaOp.getResult();

    auto kvaluesAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op.getValuesAttr());
    if (!kvaluesAttr)
      return mlir::failure();

    int64_t i = 0;
    for (const llvm::APFloat &val : kvaluesAttr.getValues<llvm::APFloat>()) {
      int64_t idx = i++;
      int64_t r = idx / cols;
      int64_t c = idx % cols;

      auto cRow = rewriter.create<mlir::arith::ConstantIndexOp>(loc, r);
      auto cCol = rewriter.create<mlir::arith::ConstantIndexOp>(loc, c);
      auto cVal = rewriter.create<mlir::arith::ConstantFloatOp>(loc, f64Type, val);

      rewriter.create<mlir::memref::StoreOp>(loc, cVal, kStack, mlir::ValueRange{cRow, cCol});
    }

    rewriter.replaceOp(op, kStack);
    return mlir::success();
  }
};

#define GEN_PASS_DEF_PICCELERKERNELTOMEMREF
#include "piccelerPasses.h.inc"

struct PiccelerKernelToMemrefPass : public impl::PiccelerKernelToMemrefBase<PiccelerKernelToMemrefPass> {
  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion([](KernelType kernelType) -> mlir::Type {
      return mlir::MemRefType::get({kernelType.getRows(), kernelType.getCols()},
                                   mlir::Float64Type::get(kernelType.getContext()));
    });

    mlir::RewritePatternSet patterns(&context);
    patterns.add<KernelToMemref>(typeConverter, &context);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

    mlir::ConversionTarget target(context);
    target.addIllegalOp<KernelConstOp>();
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::memref::MemRefDialect, mlir::func::FuncDialect>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerKernelToMemrefPass() {
  return std::make_unique<PiccelerKernelToMemrefPass>();
}

} // namespace picceler