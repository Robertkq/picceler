#pragma once

#include <mlir/Pass/PassManager.h>

#include "passes.h"

namespace picceler {

class IRPassManager {
public:
  IRPassManager(mlir::MLIRContext *context) : _passManager(context) {
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
  mlir::PassManager _passManager;
};

} // namespace picceler