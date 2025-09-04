#pragma once

#include <CLI/CLI.hpp>
#include <format>
#include <iostream>
#include <spdlog/spdlog.h>

#include "parser.h"

namespace picceler {

class App {
public:
  App();
  void test();

  CLI::App &getCliApp() { return _cliApp; }

private:
  CLI::App _cliApp;
  std::string _inputFile;
  Parser _parser;
};

} // namespace picceler
