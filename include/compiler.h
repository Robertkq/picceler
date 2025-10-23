#pragma once

#include <CLI/CLI.hpp>
#include <format>
#include <iostream>
#include <spdlog/spdlog.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>

#include "dialect.h"
#include "mlir_gen.h"
#include "parser.h"
#include "pass_manager.h"

namespace picceler {

class Compiler {
public:
  Compiler();
  void run();

  CLI::App &getCliApp() { return _cliApp; }

private:
  mlir::MLIRContext &getContext() { return _context; }
  mlir::DialectRegistry initRegistry();

private:
  CLI::App _cliApp;
  std::string _inputFile;
  Parser _parser;

  mlir::MLIRContext _context;
  MLIRGen _mlirGen;
  IRPassManager _passManager;
};

} // namespace picceler
