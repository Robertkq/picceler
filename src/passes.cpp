#include <spdlog/spdlog.h>

#include "passes.h"

namespace picceler {

void PiccelerTypesToLLVMIRPass::runOnOperation() {
  spdlog::trace("Running PiccelerTypesToLLVMIRPass");
  auto op = getOperation();
  spdlog::trace("Operation name: {}", op.getName()->data());
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