#include <vector>

#include "spdlog/spdlog.h"

#include "compiler.h"

int main(int argc, char **argv) {
  picceler::Compiler compiler;

  if (!compiler.parseArgs(argc, argv)) {
    spdlog::error("Failed to parse command-line arguments");
    return 1;
  }

  if (!compiler.run()) {
    spdlog::error("Compilation failed");
    return 1;
  }

  return 0;
}
