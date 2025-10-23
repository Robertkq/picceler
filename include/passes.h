#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/IR/BuiltinOps.h>

namespace picceler {

class LowerPiccelerOpsToFuncCallsPass
    : public mlir::PassWrapper<LowerPiccelerOpsToFuncCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override;
  mlir::StringRef getDescription() const override;
  static void registerPass();
  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<LowerPiccelerOpsToFuncCallsPass>();
  }
};

class PiccelerTypesToLLVMIRPass
    : public mlir::PassWrapper<PiccelerTypesToLLVMIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override;
  mlir::StringRef getDescription() const override;
  static void registerPass();
  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<PiccelerTypesToLLVMIRPass>();
  }
};

} // namespace picceler