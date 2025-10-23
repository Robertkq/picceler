#include "parser.h"
#include <iostream>

using namespace picceler;

Parser::Parser() : _lexer() {}

void Parser::setSource(const std::string &source) { _lexer.setSource(source); }

std::vector<Token> Parser::getTokens() { return _lexer.tokenizeAll(); }

std::unique_ptr<ModuleNode> Parser::parse() {
  spdlog::info("Starting parsing process");
  auto module = std::make_unique<ModuleNode>();
  while (true) {
    auto stmt = parseStatement();
    if (!stmt) {
      break;
    }
    spdlog::info("Parsed statement: {}", stmt->toString());
    module->statements.push_back(std::move(stmt));
  }
  spdlog::info("Finished parsing process");
  return module;
}

std::unique_ptr<ASTNode> Parser::parseStatement() {
  auto token = _lexer.peekToken();
  if (token._type == Token::Type::IDENTIFIER) {
    spdlog::debug("Parsing statement starting with identifier '{}'",
                  token._value);
    auto identifier = _lexer.nextToken();
    auto nextToken = _lexer.peekToken();
    if (nextToken._type == Token::Type::SYMBOL) {
      if (nextToken._value == "=") {
        return parseAssignment(identifier);
      } else if (nextToken._value == "(") {
        return parseCall(identifier);
      }
    }
    spdlog::error("Unexpected token '{}' at {}:{}", nextToken._value,
                  nextToken._line, nextToken._column);
    return nullptr;
  } else if (token._type == Token::Type::EOF_TOKEN) {
    return nullptr;
  } else {
    spdlog::error("Unexpected token '{}' at {}:{}", token._value, token._line,
                  token._column);
    return nullptr;
  }
}

std::unique_ptr<ASTNode> Parser::parseAssignment(Token identifier) {
  spdlog::debug("Parsing assignment for identifier '{}'", identifier._value);
  auto eqToken = _lexer.nextToken(); // consume '='
  if (eqToken._type != Token::Type::SYMBOL || eqToken._value != "=") {
    spdlog::error("Expected '=' after identifier at {}:{}", eqToken._line,
                  eqToken._column);
    return nullptr;
  }
  auto expr = parseExpression();
  if (!expr) {
    spdlog::error("Invalid expression in assignment at {}:{}", eqToken._line,
                  eqToken._column);
    return nullptr;
  }
  auto assignNode = std::make_unique<AssignmentNode>();
  assignNode->lhs = std::unique_ptr<VariableNode>(
      dynamic_cast<VariableNode *>(parseVariable(identifier).release()));
  assignNode->rhs = std::move(expr);
  return assignNode;
}

std::unique_ptr<ASTNode> Parser::parseCall(Token identifier) {
  spdlog::debug("Parsing function call for identifier '{}'", identifier._value);
  auto lparenToken = _lexer.nextToken(); // consume '('
  if (lparenToken._type != Token::Type::SYMBOL || lparenToken._value != "(") {
    spdlog::error("Expected '(' after function name at {}:{}",
                  lparenToken._line, lparenToken._column);
    return nullptr;
  }
  auto callNode = std::make_unique<CallNode>();
  callNode->callee = identifier._value;

  while (true) {
    auto nextToken = _lexer.peekToken();
    if (nextToken._type == Token::Type::SYMBOL && nextToken._value == ")") {
      _lexer.nextToken(); // consume ')'
      break;
    }
    auto arg = parseExpression();
    if (!arg) {
      spdlog::error("Invalid argument in function call at {}:{}",
                    nextToken._line, nextToken._column);
      return nullptr;
    }
    callNode->arguments.push_back(std::move(arg));

    nextToken = _lexer.peekToken();
    if (nextToken._type == Token::Type::SYMBOL && nextToken._value == ",") {
      _lexer.nextToken(); // consume ','
    } else if (nextToken._type == Token::Type::SYMBOL &&
               nextToken._value == ")") {
      continue;
    } else {
      spdlog::error("Expected ',' or ')' in function call at {}:{}",
                    nextToken._line, nextToken._column);
      return nullptr;
    }
  }

  return callNode;
}

std::unique_ptr<ASTNode> Parser::parseExpression() {
  spdlog::debug("Parsing expression");
  auto token = _lexer.peekToken();
  if (token._type == Token::Type::IDENTIFIER) {
    auto identifier = _lexer.nextToken();
    auto nextToken = _lexer.peekToken();
    if (nextToken._type == Token::Type::SYMBOL && nextToken._value == "(") {
      return parseCall(identifier);
    } else {
      return parseVariable(identifier);
    }
  } else if (token._type == Token::Type::STRING) {
    return parseString();
  } else if (token._type == Token::Type::NUMBER) {
    return parseNumber();
  } else {
    spdlog::error("Unexpected token '{}' at {}:{}", token._value, token._line,
                  token._column);
    return nullptr;
  }
}

std::unique_ptr<ASTNode> Parser::parseVariable(Token identifier) {
  spdlog::debug("Parsing variable");
  // If the passed token is not an identifier, consume the next token
  if (identifier._type != Token::Type::IDENTIFIER) {
    identifier = _lexer.nextToken();
  }

  auto varNode = std::make_unique<VariableNode>();
  varNode->name = identifier._value;
  return varNode;
}

std::unique_ptr<ASTNode> Parser::parseString() {
  spdlog::debug("Parsing string");
  auto token = _lexer.nextToken(); // consume string
  if (token._type != Token::Type::STRING) {
    spdlog::error("Expected string at {}:{}", token._line, token._column);
    return nullptr;
  }
  auto strNode = std::make_unique<StringNode>();
  strNode->value = token._value;
  return strNode;
}

std::unique_ptr<ASTNode> Parser::parseNumber() {
  spdlog::debug("Parsing number");
  auto token = _lexer.nextToken(); // consume number
  if (token._type != Token::Type::NUMBER) {
    spdlog::error("Expected number at {}:{}", token._line, token._column);
    return nullptr;
  }
  auto numNode = std::make_unique<NumberNode>();
  numNode->value = std::stoul(token._value);
  return numNode;
}

void Parser::printAST(const std::unique_ptr<ModuleNode> &node, int indent) {
  if (!node)
    return;
  std::string indentStr(indent, ' ');
  spdlog::info("{}", node->toString());
  for (const auto &stmt : node->statements) {
    spdlog::info("{}", stmt->toString());
  }
}
