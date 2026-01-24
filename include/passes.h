#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace picceler {

/**
 * @brief Pass that lowers Picceler operations to function calls.
 */
class LowerPiccelerOpsToFuncCallsPass
    : public mlir::PassWrapper<LowerPiccelerOpsToFuncCallsPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override { return "lower-picceler-ops-to-func-calls"; }
  mlir::StringRef getDescription() const override { return "Lower Picceler operations to function calls"; }
  static void registerPass() { mlir::PassRegistration<LowerPiccelerOpsToFuncCallsPass>(); }
  static std::unique_ptr<mlir::Pass> create() { return std::make_unique<LowerPiccelerOpsToFuncCallsPass>(); }
};

/**
 * @brief Pass that lowers named Picceler filters (gaussian_blur, sharpen, etc.) to explicit convolution operations.
 */
class LowerPiccelerFiltersToConvPass
    : public mlir::PassWrapper<LowerPiccelerFiltersToConvPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override { return "lower-picceler-filters-to-conv"; }
  mlir::StringRef getDescription() const override {
      return "Lower named Picceler filters to explicit convolutions";
  }
  static void registerPass() { mlir::PassRegistration<LowerPiccelerFiltersToConvPass>(); }
  static std::unique_ptr<mlir::Pass> create() { return std::make_unique<LowerPiccelerFiltersToConvPass>(); }
};

/**
 * @brief Pass that converts Picceler types to LLVM IR types.
 */
/**
 * @brief Unified pass that converts Picceler types, constants, and functions to
 * LLVM IR. This is the final lowering step for the Picceler dialect.
 */
class PiccelerToLLVMConversionPass
    : public mlir::PassWrapper<PiccelerToLLVMConversionPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override { return "convert-picceler-to-llvm"; }
  mlir::StringRef getDescription() const override {
    return "Convert Picceler dialect types and residual ops to LLVM dialect";
  }
  static void registerPass() { mlir::PassRegistration<PiccelerToLLVMConversionPass>(); }
  static std::unique_ptr<mlir::Pass> create() { return std::make_unique<PiccelerToLLVMConversionPass>(); }
};

/**
 * @brief Pass that converts Picceler operations to Affine dialect.
 */
class PiccelerToAffinePass : public mlir::PassWrapper<PiccelerToAffinePass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override { return "convert-picceler-to-affine"; }
  mlir::StringRef getDescription() const override { return "Convert Picceler operations to Affine dialect"; }
  static void registerPass() { mlir::PassRegistration<PiccelerToAffinePass>(); }
  static std::unique_ptr<mlir::Pass> create() { return std::make_unique<PiccelerToAffinePass>(); }
};

/**
 * @brief Pass that converts Picceler ConstKernelOps to memref dialect.
 */

class PiccelerKernelToMemrefPass
    : public mlir::PassWrapper<PiccelerKernelToMemrefPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override;
  mlir::StringRef getArgument() const override { return "convert-picceler-kernel-to-memref"; }
  mlir::StringRef getDescription() const override {
    return "Convert Picceler KernelConstOp operations to Memref dialect";
  }
  static void registerPass() { mlir::PassRegistration<PiccelerKernelToMemrefPass>(); }
  static std::unique_ptr<mlir::Pass> create() { return std::make_unique<PiccelerKernelToMemrefPass>(); }
};

} // namespace picceler
