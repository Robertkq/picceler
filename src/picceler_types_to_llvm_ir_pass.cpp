#include <spdlog/spdlog.h>

#include "passes.h"
#include "ops.h"
#include "types.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>

namespace picceler {

void PiccelerTypesToLLVMIRPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = &getContext();

  mlir::LLVMTypeConverter converter(ctx);

  converter.addConversion([ctx](picceler::StringType) -> mlir::Type {
      return mlir::LLVM::LLVMPointerType::get(ctx, 8);
  });

  converter.addConversion([&](picceler::ImageType) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(ctx, 8);
  });

  mlir::RewritePatternSet patterns(ctx);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<picceler::StringConstOp>();

  if (failed(applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

mlir::StringRef PiccelerTypesToLLVMIRPass::getArgument() const {
  return "picceler-types-to-llvmir";
}

mlir::StringRef PiccelerTypesToLLVMIRPass::getDescription() const {
  return "Convert Picceler types to LLVM IR types";
}

void PiccelerTypesToLLVMIRPass::registerPass() {
  mlir::PassRegistration<PiccelerTypesToLLVMIRPass>();
}

} // namespace picceler
