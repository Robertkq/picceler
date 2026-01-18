#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

#include "ops.h"
#include "types.h"

namespace picceler {

struct KernelToMemref : mlir::OpConversionPattern<KernelConstOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(KernelConstOp op, KernelConstOpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter rewriter) {
    return mlir::success();
  }
};

void PiccelerKernelToMemrefPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = &getContext();
  mlir::Location loc = module.getLoc();

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<KernelToMemref>(ctx);
}
} // namespace picceler