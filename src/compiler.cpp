#include "compiler.h"
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

namespace picceler {

Compiler::Compiler()
    : _cliApp("picceler compiler"), _inputFile(), _parser(),
      _context(initRegistry()), _mlirGen(&_context), _passManager(&_context) {
  spdlog::cfg::load_env_levels();
  _cliApp.add_option("-i,--input", _inputFile, "Input source file")->required();
  _context.loadDialect<picceler>();
  _context.loadDialect<mlir::func::FuncDialect>();
  _context.loadDialect<mlir::arith::ArithDialect>();
  for (auto *dialect : _context.getLoadedDialects()) {
    llvm::outs() << dialect->getNamespace() << "\n";
  }
}

mlir::DialectRegistry Compiler::initRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<picceler>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  return registry;
}

void Compiler::run() {
  _parser.setSource(_inputFile);
  spdlog::info("Tokenizing source file: {}", _inputFile);
  auto tokens = _parser.getTokens();
  size_t index = 0;
  for (const auto &token : tokens) {
    spdlog::info("Token[{}]: {}", index++, token.toString());
  }
  spdlog::info("Finished tokenizing source file");
  spdlog::info("Resetting lexer");
  _parser.setSource(_inputFile);
  auto ast = _parser.parse();
  _parser.printAST(ast);

  spdlog::info("Generating initial MLIR");
  auto module = _mlirGen.generate(ast.get());
  spdlog::info("Finished generating initial MLIR");
  spdlog::info("Running pass manager");
  _passManager.run(module);
  spdlog::info("Finished running pass manager");
}

} // namespace picceler