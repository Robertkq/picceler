#include "pass_manager.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

namespace picceler {

IRPassManager::IRPassManager(mlir::MLIRContext *context)
    : _passManager(context) {
  std::error_code ec;
  _outStream = std::make_unique<llvm::raw_fd_ostream>("output.mlir", ec,
                                                      llvm::sys::fs::OF_Text);
  if (ec) {
    throw std::runtime_error("Could not open output.mlir: " + ec.message());
  }
  _outStream->SetUnbuffered();
  _passManager.enableVerifier();
  _passManager.enableIRPrinting(
      [](mlir::Pass *, mlir::Operation *) { return false; },
      [](mlir::Pass *, mlir::Operation *) { return true; }, false, false, false,
      *_outStream);
  _passManager.enableIRPrintingToFileTree(
      nullptr, [](mlir::Pass *, mlir::Operation *) { return true; }, false,
      false);
  _passManager.addInstrumentation(std::make_unique<PassLogger>());
  registerPasses();
  addPasses();
};

void IRPassManager::run(mlir::ModuleOp module) {
  if (mlir::failed(_passManager.run(module))) {
    throw std::runtime_error("Failed to run pass manager");
  }
}

void IRPassManager::registerPasses() {
  PiccelerToAffinePass::registerPass();
  LowerPiccelerOpsToFuncCallsPass::registerPass();
  PiccelerToLLVMConversionPass::registerPass();
}

void IRPassManager::addPasses() {
  addHighLevelOptimizationPasses();
  addAffineLoweringPasses();
  addRuntimeLoweringPasses();
  addBackendLoweringPasses();
}

void IRPassManager::addHighLevelOptimizationPasses() {}

void IRPassManager::addAffineLoweringPasses() {
  _passManager.addPass(mlir::createCanonicalizerPass());
  _passManager.addPass(PiccelerToAffinePass::create());
}
void IRPassManager::addRuntimeLoweringPasses() {

  _passManager.addPass(LowerPiccelerOpsToFuncCallsPass::create());
}
void IRPassManager::addBackendLoweringPasses() {
  _passManager.addPass(PiccelerToLLVMConversionPass::create());
  _passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
  _passManager.addPass(mlir::createCanonicalizerPass());
  _passManager.addPass(mlir::createLowerAffinePass());
  _passManager.addPass(mlir::createSCFToControlFlowPass());
  _passManager.addPass(mlir::createArithToLLVMConversionPass());
  _passManager.addPass(mlir::createConvertControlFlowToLLVMPass());
  _passManager.addPass(mlir::createConvertFuncToLLVMPass());
  _passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

} // namespace picceler