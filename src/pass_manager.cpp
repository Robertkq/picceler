#include "pass_manager.h"

#include "mlir/Transforms/Passes.h"

namespace picceler {

IRPassManager::IRPassManager(mlir::MLIRContext *context)
    : _passManager(context) {
  std::error_code ec;
  _outStream = std::make_unique<llvm::raw_fd_ostream>("output.mlir", ec,
                                                      llvm::sys::fs::OF_Text);
  if (ec) {
    throw std::runtime_error("Could not open output.mlir: " + ec.message());
  }
  _passManager.enableVerifier();
  _passManager.enableIRPrinting(
      [](mlir::Pass *, mlir::Operation *) { return true; },
      [](mlir::Pass *, mlir::Operation *) { return true; }, false, false, false,
      *_outStream);
  _passManager.enableIRPrintingToFileTree(
      [](mlir::Pass *, mlir::Operation *) { return true; },
      [](mlir::Pass *, mlir::Operation *) { return true; }, false, false);
  registerPasses();
  addPasses();
};

void IRPassManager::run(mlir::ModuleOp module) {
  if (mlir::failed(_passManager.run(module))) {
    throw std::runtime_error("Failed to run pass manager");
  }
}

void IRPassManager::registerPasses() {
  LowerPiccelerOpsToFuncCallsPass::registerPass();
  PiccelerTypesToLLVMIRPass::registerPass();
  PiccelerConstOpsToLLVMIRPass::registerPass();
}

void IRPassManager::addPasses() {
  _passManager.addPass(LowerPiccelerOpsToFuncCallsPass::create());
  _passManager.addPass(PiccelerTypesToLLVMIRPass::create());
  _passManager.addPass(PiccelerConstOpsToLLVMIRPass::create());
  _passManager.addPass(mlir::createCanonicalizerPass());
}

} // namespace picceler