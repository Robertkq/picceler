#pragma once

#include <format>
#include <iostream>
#include <memory>

#include "llvm/Support/CommandLine.h"
#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/Module.h"

#include "dialect.h"
#include "mlir_gen.h"
#include "parser.h"
#include "pass_manager.h"

namespace picceler {

/**
 * @brief Struct to hold command-line options for the compiler.
 */
class CLIOptions {
public:
  std::string inputFile;
  std::string outputFile;
};

/**
 * @brief Compiler class that orchestrates the compilation process.
 */
class Compiler {
public:
  Compiler();

  /**
   * @brief Parses command-line arguments.
   * @return True if parsing was successful, false otherwise.
   */
  bool parseArgs(int argc, char **argv);

  /**
   * @brief Runs the compilation process.
   */
  bool run();

  const CLIOptions &getCliOptions() const { return _cliOptions; }

private:
  mlir::MLIRContext &getContext();

  /**
   * @brief Initializes the MLIR dialect registry with the necessary dialects for the compiler.
   */
  static mlir::DialectRegistry initRegistry();

  /**
   * @brief Emits an object file from the given LLVM module.
   * @param llvmModule The LLVM module to emit.
   * @param objFilename The name of the output object file.
   * @return True if the object file was emitted successfully, false otherwise.
   */
  bool emitObjectFile(llvm::Module *llvmModule, const std::string &objFilename);

  /**
   * @brief Links the generated object file with the runtime library to produce the final executable.
   * @param objFile The object file to link.
   * @param runtimeLib The runtime library to link against.
   * @param outputExe The name of the output executable file.
   * @return True if the linking was successful, false otherwise.
   */
  bool linkWithLLD(const std::string &objFile, const std::string &runtimeLib, const std::string &outputExe);

private:
  CLIOptions _cliOptions;
  Parser _parser;

  // Lazily initialized to avoid LLVM option registration conflicts
  std::unique_ptr<mlir::MLIRContext> _context;
  std::unique_ptr<MLIRGen> _mlirGen;
  std::unique_ptr<IRPassManager> _passManager;
};

} // namespace picceler
