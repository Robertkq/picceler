#pragma once

#include <mlir/Pass/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>

#include "passes.h"

namespace picceler {

class IRPassManager {
public:
  IRPassManager(mlir::MLIRContext *context) : _passManager(context) {
    std::error_code ec;
    _outStream = std::make_unique<llvm::raw_fd_ostream>("output.mlir", ec,
                                                        llvm::sys::fs::OF_Text);
    if (ec) {
      throw std::runtime_error("Could not open output.mlir: " + ec.message());
    }
    _passManager.enableVerifier();
    _passManager.enableIRPrinting(
        [](mlir::Pass *, mlir::Operation *) { return true; },
        [](mlir::Pass *, mlir::Operation *) { return true; }, false, false,
        false, *_outStream);
    _passManager.enableIRPrintingToFileTree(
        [](mlir::Pass *, mlir::Operation *) { return true; },
        [](mlir::Pass *, mlir::Operation *) { return true; }, false, false);
    registerPasses();
    addPasses();
  };

  void run(mlir::ModuleOp module) {
    if (mlir::failed(_passManager.run(module))) {
      throw std::runtime_error("Failed to run pass manager");
    }
  }

private:
  void registerPasses() {
    // Register custom passes here
    LowerPiccelerOpsToFuncCallsPass::registerPass();
    PiccelerTypesToLLVMIRPass::registerPass();
  }

  void addPasses() {
    // Add passes to the pass manager here
    _passManager.addPass(LowerPiccelerOpsToFuncCallsPass::create());
    _passManager.addPass(PiccelerTypesToLLVMIRPass::create());
  }

private:
  std::unique_ptr<llvm::raw_fd_ostream> _outStream;
  mlir::PassManager _passManager;
};

} // namespace picceler