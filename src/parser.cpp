#include "parser.h"

using namespace picceler;

Parser::Parser() : _lexer() {}

void Parser::setSource(const std::string &source) { _lexer.setSource(source); }

std::vector<Token> Parser::getTokens() { return _lexer.tokenizeAll(); }