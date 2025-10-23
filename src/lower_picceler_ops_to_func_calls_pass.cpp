#include <spdlog/spdlog.h>

#include "passes.h"
#include "ops.h"
#include "types.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/SymbolTable.h>

namespace picceler {

void LowerPiccelerOpsToFuncCallsPass::runOnOperation() {
  spdlog::trace("Running LowerPiccelerOpsToFuncCallsPass");
  mlir::ModuleOp module = getOperation();

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<LoadImageToCall>(&getContext());
  patterns.add<BlurImageToCall>(&getContext());
  patterns.add<ShowImageToCall>(&getContext());
  patterns.add<SaveImageToCall>(&getContext());

  if (mlir::failed(
          mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

mlir::StringRef LowerPiccelerOpsToFuncCallsPass::getArgument() const {
  return "lower-picceler-ops-to-func-calls";
}

mlir::StringRef LowerPiccelerOpsToFuncCallsPass::getDescription() const {
  return "Lower Picceler operations to function calls";
}

void LowerPiccelerOpsToFuncCallsPass::registerPass() {
  mlir::PassRegistration<LowerPiccelerOpsToFuncCallsPass>();
}

} // namespace picceler