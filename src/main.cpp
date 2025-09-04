#include "app.h"
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <vector>

int main(int argc, char **argv) {
  picceler::App app;
  CLI11_PARSE(app.getCliApp(), argc, argv);

  app.test();
}
