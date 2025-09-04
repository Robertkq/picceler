#include "app.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <vector>

int main(int argc, char **argv) {
  CLI::App cliApp{"Picceler Compiler"};
  std::string inputFile;
  cliApp.add_option("-i,--input", inputFile, "Input source file")->required();
  CLI11_PARSE(cliApp, argc, argv);

  picceler::App app(inputFile);
  app.test();
}
