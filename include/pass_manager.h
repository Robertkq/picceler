#pragma once

#include <memory>

#include "spdlog/spdlog.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace picceler {

/**
 * @brief Custom struct to log pass execution using spdlog.
 * This struct implements the PassInstrumentation interface to hook into the pass execution lifecycle.
 */
struct PassLogger : public mlir::PassInstrumentation {
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    spdlog::info("Started pass: {}", pass->getName().str());
  }
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    spdlog::info("Finished pass: {}", pass->getName().str());
  }
  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    spdlog::error("Failed pass: {}", pass->getName().str());
  }
};

/**
 * @brief Wrapper around MLIR PassManager to manage and run passes.
 */
class IRPassManager {
public:
  /**
   * @brief Constructs an IRPassManager with the given MLIR context.
   * Also sets up the output stream and configures IR printing.
   * @param context The MLIR context to use.
   */
  IRPassManager(mlir::MLIRContext *context);

  /**
   * @brief Runs the pass manager on the given MLIR module.
   * @param module The MLIR module to run the passes on.
   */
  void run(mlir::ModuleOp module);

private:
  /**
   * @brief This function adds all the passes to the pass manager.
   */
  void addPasses();

  /**
   * @name The following functions describe the phases of compilation
   * and groups passes accordingly.
   * @{
   */

  /**
   *  @brief Add passes for the high-level optimization phase.
   *
   */
  void addHighLevelOptimizationPasses();

  /**
   * @brief Add passes that lower the IR toward affine/loop forms.
   */
  void addAffineLoweringPasses();

  /**
   * @brief Add passes that replace runtime-level ops with runtime calls.
   */
  void addRuntimeLoweringPasses();

  /**
   *  @brief Add passes for final backend lowering.
   */
  void addBackendLoweringPasses();

  /** @} */

private:
  std::unique_ptr<llvm::raw_fd_ostream> _outStream;
  mlir::PassManager _passManager;
};

} // namespace picceler