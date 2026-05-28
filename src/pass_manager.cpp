/**
 * @file pass_manager.cpp
 * @brief Implementation of the IRPassManager class, which manages the sequence of MLIR passes
 * that transform the MLIR module from its initial form to the final LLVM IR form ready for code generation. The pass
 * manager is responsible for organizing passes into logical phases of compilation, such as high-level optimization,
 * affine lowering, and backend lowering. It also sets up instrumentation to log the execution of passes for debugging
 * and performance analysis.
 */

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

IRPassManager::IRPassManager(mlir::MLIRContext *context) : _passManager(context) {
  std::error_code ec;
  std::string outputFilename = "dump.mlir";
  _outStream = std::make_unique<llvm::raw_fd_ostream>(outputFilename, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    throw std::runtime_error("Could not open " + outputFilename + ": " + ec.message());
  }
  _outStream->SetUnbuffered();
  _passManager.enableVerifier();
  _passManager.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return false; },
                                [](mlir::Pass *, mlir::Operation *) { return true; }, false, false, false, *_outStream);
  _passManager.enableIRPrintingToFileTree(nullptr, [](mlir::Pass *, mlir::Operation *) { return true; }, false, false);
  _passManager.addInstrumentation(std::make_unique<PassLogger>());
  addPasses();
};

void IRPassManager::run(mlir::ModuleOp module) {
  if (mlir::failed(_passManager.run(module))) {
    throw std::runtime_error("Failed to run pass manager");
  }
}

void IRPassManager::addPasses() {
  addRuntimeLoweringPasses();
  addHighLevelOptimizationPasses();
  addAffineLoweringPasses();
  addBackendLoweringPasses();
}

void IRPassManager::addRuntimeLoweringPasses() { _passManager.addPass(createPiccelerOpsToFuncCallsPass()); }

void IRPassManager::addHighLevelOptimizationPasses() {
  _passManager.addPass(mlir::createCanonicalizerPass());
  _passManager.addPass(createPiccelerFiltersToConvPass());
}

void IRPassManager::addAffineLoweringPasses() {
  _passManager.addPass(createPiccelerKernelToMemrefPass());
  _passManager.addPass(createPiccelerToAffinePass());
}

void IRPassManager::addBackendLoweringPasses() {

  _passManager.addPass(createPiccelerToLLVMIRPass());
  _passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
  _passManager.addPass(mlir::createCanonicalizerPass());
  _passManager.addPass(mlir::createLowerAffinePass());
  _passManager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  _passManager.addPass(mlir::createSCFToControlFlowPass());
  _passManager.addPass(mlir::createArithToLLVMConversionPass());
  _passManager.addPass(mlir::createConvertControlFlowToLLVMPass());
  _passManager.addPass(mlir::createConvertFuncToLLVMPass());
  _passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

} // namespace picceler