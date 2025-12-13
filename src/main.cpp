#include <vector>

#include "CLI/CLI.hpp"
#include "spdlog/spdlog.h"

#include "compiler.h"

int main(int argc, char **argv) {
  picceler::Compiler compiler;
  CLI11_PARSE(compiler.getCliApp(), argc, argv);

  compiler.run();
}
