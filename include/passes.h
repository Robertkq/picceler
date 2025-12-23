#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace picceler {

/**
 * @brief Pass that lowers Picceler operations to function calls.
 */
class LowerPiccelerOpsToFuncCallsPass
    : public mlir::PassWrapper<LowerPiccelerOpsToFuncCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override {
    return "lower-picceler-ops-to-func-calls";
  }
  mlir::StringRef getDescription() const override {
    return "Lower Picceler operations to function calls";
  }
  static void registerPass() {
    mlir::PassRegistration<LowerPiccelerOpsToFuncCallsPass>();
  }
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
  mlir::StringRef getArgument() const override {
    return "convert-picceler-types-to-llvm-ir";
  }
  mlir::StringRef getDescription() const override {
    return "Convert Picceler types to LLVM IR types";
  }
  static void registerPass() {
    mlir::PassRegistration<PiccelerTypesToLLVMIRPass>();
  }
  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<PiccelerTypesToLLVMIRPass>();
  }
};

class PiccelerConstOpsToLLVMIRPass
    : public mlir::PassWrapper<PiccelerConstOpsToLLVMIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override {
    return "convert-picceler-const-ops-to-llvm-ir";
  }
  mlir::StringRef getDescription() const override {
    return "Convert Picceler constant operations to LLVM IR";
  }
  static void registerPass() {
    mlir::PassRegistration<PiccelerConstOpsToLLVMIRPass>();
  }
  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<PiccelerConstOpsToLLVMIRPass>();
  }
};

class PiccelerToAffinePass
    : public mlir::PassWrapper<PiccelerToAffinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override {
    return "convert-picceler-to-affine";
  }
  mlir::StringRef getDescription() const override {
    return "Convert Picceler operations to Affine dialect";
  }
  static void registerPass() { mlir::PassRegistration<PiccelerToAffinePass>(); }
  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<PiccelerToAffinePass>();
  }
};

} // namespace picceler