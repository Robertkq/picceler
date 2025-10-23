#include "compiler.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <vector>

int main(int argc, char **argv) {
  picceler::Compiler compiler;
  CLI11_PARSE(compiler.getCliApp(), argc, argv);

  compiler.run();
}
