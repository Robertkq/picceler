#include "parser.h"

using namespace picceler;

Parser::Parser(const std::string &source) : _lexer(source) {}

std::vector<Token> Parser::getTokens() { return _lexer.tokenizeAll(); }