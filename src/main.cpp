#include <vector>

#include "CLI/CLI.hpp"
#include "spdlog/spdlog.h"

#include "compiler.h"

int main(int argc, char **argv) {
  std::cout << "staart\n";
  picceler::Compiler compiler;
  std::cout << "after compiler\n";
  CLI11_PARSE(compiler.getCliApp(), argc, argv);

  compiler.run();
}
