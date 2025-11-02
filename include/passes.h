#pragma once

#include <mlir/Pass/Pass.h>
#include <mlir/IR/BuiltinOps.h>

namespace picceler {

/**
 * @brief Pass that lowers Picceler operations to function calls.
 */
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

/**
 * @brief Pass that converts Picceler types to LLVM IR types.
 */
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