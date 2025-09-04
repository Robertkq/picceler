#pragma once

#include "lexer.h"

namespace picceler {

class Parser {
public:
  Parser(const std::string &source);
  std::vector<Token> getTokens();

private:
  Lexer _lexer;
};

} // namespace picceler