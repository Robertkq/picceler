#include "app.h"

using namespace picceler;

App::App(const std::string &source) : _parser(source) {
  spdlog::info("App initialized with source file: {}", source);
}

void App::test() {
  auto tokens = _parser.getTokens();
  for (const auto &token : tokens) {
    std::cout << token << "\n";
  }
}