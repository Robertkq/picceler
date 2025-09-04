#pragma once

#include <format>
#include <iostream>
#include <spdlog/spdlog.h>

#include "parser.h"

namespace picceler {

class App {
public:
  App(const std::string &source);
  void test();

private:
  Parser _parser;
};

} // namespace picceler