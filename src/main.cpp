#include "CLI/CLI.hpp"
#include "spdlog/cfg/env.h"

#include "compiler.h"

int main(int argc, char **argv) {
  spdlog::cfg::load_env_levels();
  picceler::Compiler compiler;
  CLI11_PARSE(compiler.getCliApp(), argc, argv);

  compiler.run();
}
