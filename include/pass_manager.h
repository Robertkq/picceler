#pragma once

#include <mlir/Pass/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>

#include "passes.h"

namespace picceler {

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

  /*
   * @brief Runs the pass manager on the given MLIR module.
   * @param module The MLIR module to run the passes on.
   */
  void run(mlir::ModuleOp module);

private:
  /**
   * @brief Registers all necessary passes.
   * To be used for mlir-opt compatibility.
   */
  void registerPasses();

  /**
   * @brief This function adds the passes to the pass manager.
   */
  void addPasses();

private:
  std::unique_ptr<llvm::raw_fd_ostream> _outStream;
  mlir::PassManager _passManager;
};

} // namespace picceler