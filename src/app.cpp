#include "app.h"
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

namespace picceler {

App::App() : _cliApp("picceler compiler"), _inputFile(), _parser() {
  spdlog::cfg::load_env_levels();
  _cliApp.add_option("-i,--input", _inputFile, "Input source file")->required();
}

void App::run() {
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
}

} // namespace picceler