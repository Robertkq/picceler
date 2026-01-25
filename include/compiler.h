#pragma once

#include <format>
#include <iostream>

#include "CLI/CLI.hpp"
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
   * @brief Runs the compilation process.
   */
  bool run();

  CLI::App &getCliApp() { return _cliApp; }
  const CLIOptions &getCliOptions() const { return _cliOptions; }

private:
  mlir::MLIRContext &getContext() { return _context; }
  static mlir::DialectRegistry initRegistry();

  bool emitObjectFile(llvm::Module *llvmModule, const std::string &objFilename);
  bool linkWithLLD(const std::string &objFile, const std::string &runtimeLib, const std::string &outputExe);

private:
  CLI::App _cliApp;
  CLIOptions _cliOptions;
  Parser _parser;

  mlir::MLIRContext _context;
  MLIRGen _mlirGen;
  IRPassManager _passManager;
};

} // namespace picceler
