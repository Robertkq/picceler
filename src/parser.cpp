#include "parser.h"

#include <format>
#include <stdexcept>
#include <vector>

#include "spdlog/spdlog.h"
#include "error.h"
#include "utils.h"

using namespace picceler;

Parser::Parser() : _tokens(), _index(0) {}

Parser::Parser(std::vector<Token> tokens) : _tokens(std::move(tokens)), _index(0) {}

Result<std::unique_ptr<ModuleNode>> Parser::parse() {
  if (_tokens.empty() || _tokens.back().type() != Token::Type::EOF_TOKEN) {
    return std::unexpected(CompileError{"Invalid token stream: expected non-empty stream terminating in EOF"});
  }

  spdlog::debug("Starting parsing process");
  auto module = std::make_unique<ModuleNode>(_tokens.front().location());

  while (!isAtEnd()) {
    auto stmt = parseStatement();
    if (!stmt) {
      return std::unexpected(stmt.error());
    }
    module->addStatement(std::move(*stmt));
  }

  return module;
}

void Parser::printAST(const std::unique_ptr<ModuleNode> &node, int indent) {
  if (!node)
    return;
  spdlog::debug("{}", node->toString());
  for (const auto &stmt : node->statements()) {
    spdlog::debug("{}", stmt->toString());
  }
}

const Token &Parser::peek() const { return _tokens[std::min(_index, _tokens.size() - 1)]; }

const Token &Parser::peekAhead(std::size_t offset) const {
  return _tokens[std::min(_index + offset, _tokens.size() - 1)];
}

bool Parser::isAtEnd() const { return peek().type() == Token::Type::EOF_TOKEN; }

bool Parser::check(Token::Type type) const { return peek().type() == type; }

const Token &Parser::advance() {
  const Token &current = peek();
  if (!isAtEnd()) {
    _index++;
  }
  return current;
}

bool Parser::match(Token::Type type) {
  if (check(type)) {
    advance();
    return true;
  }
  return false;
}

Result<Token> Parser::consume(Token::Type type, std::string_view errorMessage) {
  if (check(type)) {
    return advance();
  }
  return std::unexpected(CompileError{std::string(errorMessage), peek().location()});
}

Result<std::unique_ptr<ASTNode>> Parser::parseStatement() {
  if (match(Token::Type::KW_DEF)) {
    return parseFunctionDefinition();
  }

  if (match(Token::Type::KW_IF)) {
    return parseIfStatement();
  }

  if (check(Token::Type::IDENTIFIER)) {
    const auto &identifier = advance();
    if (check(Token::Type::ASSIGN)) {
      return parseAssignment(identifier);
    }
    if (check(Token::Type::L_PAREN)) {
      return parseCall(identifier);
    }
    return std::unexpected(
        CompileError{std::format("Unexpected token '{}' after identifier", peek().value()), peek().location()});
  }

  return std::unexpected(CompileError{std::format("Unexpected token '{}'", peek().value()), peek().location()});
}

Result<std::unique_ptr<ASTNode>> Parser::parseFunctionDefinition() {
  auto nameTok = consume(Token::Type::IDENTIFIER, "Expected function name after 'def'");
  if (!nameTok)
    return std::unexpected(nameTok.error());

  auto funcNode = std::make_unique<FunctionNode>(nameTok->location(), nameTok->value());

  if (auto lparen = consume(Token::Type::L_PAREN, "Expected '(' in function definition"); !lparen) {
    return std::unexpected(lparen.error());
  }

  // Parse parameters
  while (!check(Token::Type::R_PAREN) && !isAtEnd()) {
    auto paramTok = consume(Token::Type::IDENTIFIER, "Expected parameter name");
    if (!paramTok)
      return std::unexpected(paramTok.error());

    if (auto colon = consume(Token::Type::COLON, "Expected ':' after parameter name"); !colon) {
      return std::unexpected(colon.error());
    }

    auto typeTok = consume(Token::Type::TYPE, "Expected type after ':'");
    if (!typeTok)
      return std::unexpected(typeTok.error());

    funcNode->addParameter(paramTok->value(), typeTok->value());

    if (!match(Token::Type::COMMA)) {
      break;
    }
  }

  if (auto rparen = consume(Token::Type::R_PAREN, "Expected ')' after parameters"); !rparen) {
    return std::unexpected(rparen.error());
  }

  if (auto lbrace = consume(Token::Type::L_BRACE, "Expected '{' before function body"); !lbrace) {
    return std::unexpected(lbrace.error());
  }

  // Parse function body
  while (!check(Token::Type::R_BRACE) && !isAtEnd()) {
    auto stmt = parseStatement();
    if (!stmt)
      return std::unexpected(stmt.error());
    funcNode->addBodyStatement(std::move(*stmt));
  }

  if (auto rbrace = consume(Token::Type::R_BRACE, "Expected '}' after function body"); !rbrace) {
    return std::unexpected(rbrace.error());
  }

  return funcNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseAssignment(const Token &identifier) {
  spdlog::debug("Parsing assignment for identifier '{}'", identifier.value());

  if (auto eqTok = consume(Token::Type::ASSIGN, "Expected '=' after identifier"); !eqTok) {
    return std::unexpected(eqTok.error());
  }

  auto exprResult = parseExpression();
  if (!exprResult)
    return std::unexpected(exprResult.error());

  auto leftVarResult = parseVariable(identifier);
  if (!leftVarResult)
    return std::unexpected(leftVarResult.error());

  auto leftVar = std::move(*leftVarResult);
  if (dynamic_cast<VariableNode *>(leftVar.get()) == nullptr) {
    return std::unexpected(CompileError{"Left-hand side of assignment must be a variable", identifier.location()});
  }

  auto leftVarNode = std::unique_ptr<VariableNode>(static_cast<VariableNode *>(leftVar.release()));
  return std::make_unique<AssignmentNode>(leftVarNode->location(), std::move(leftVarNode), std::move(*exprResult));
}

Result<std::unique_ptr<ASTNode>> Parser::parseCall(const Token &identifier) {
  spdlog::debug("Parsing function call for identifier '{}'", identifier.value());

  if (auto lparen = consume(Token::Type::L_PAREN, "Expected '(' after function name"); !lparen) {
    return std::unexpected(lparen.error());
  }

  auto callNode = std::make_unique<CallNode>(identifier.location(), identifier.value());

  while (!check(Token::Type::R_PAREN) && !isAtEnd()) {
    auto argResult = parseExpression();
    if (!argResult)
      return std::unexpected(argResult.error());

    callNode->addArgument(std::move(*argResult));

    if (!match(Token::Type::COMMA)) {
      break;
    }
  }

  if (auto rparen = consume(Token::Type::R_PAREN, "Expected ')' after arguments"); !rparen) {
    return std::unexpected(rparen.error());
  }

  return callNode;
}

// ==========================================

Result<std::unique_ptr<ASTNode>> Parser::parseIfStatement() {
  spdlog::debug("Parsing if statement");
  Location loc = peek().location();

  if (auto lparen = consume(Token::Type::L_PAREN, "Expected '(' after 'if'"); !lparen)
    return std::unexpected(lparen.error());

  auto condResult = parseExpression();
  if (!condResult)
    return std::unexpected(condResult.error());

  if (auto rparen = consume(Token::Type::R_PAREN, "Expected ')' after if condition"); !rparen)
    return std::unexpected(rparen.error());

  if (auto lbrace = consume(Token::Type::L_BRACE, "Expected '{' to open if body"); !lbrace)
    return std::unexpected(lbrace.error());

  std::vector<std::unique_ptr<ASTNode>> body;
  while (!check(Token::Type::R_BRACE) && !isAtEnd()) {
    auto stmt = parseStatement();
    if (!stmt)
      return std::unexpected(stmt.error());
    body.push_back(std::move(*stmt));
  }

  if (auto rbrace = consume(Token::Type::R_BRACE, "Expected '}' to close if body"); !rbrace)
    return std::unexpected(rbrace.error());

  return std::make_unique<IfNode>(loc, std::move(*condResult), std::move(body));
}

Result<std::unique_ptr<ASTNode>> Parser::parseExpression() {
  spdlog::debug("Parsing expression (Relational level)");
  return parseRelational();
}

Result<std::unique_ptr<ASTNode>> Parser::parseRelational() {
  auto lhs = parseAdditive();
  if (!lhs)
    return lhs;

  while (check(Token::Type::EQ) || check(Token::Type::NE) || check(Token::Type::LT) || check(Token::Type::GT) ||
         check(Token::Type::LE) || check(Token::Type::GE)) {
    Token opTok = advance();
    auto rhs = parseAdditive();
    if (!rhs)
      return rhs;

    Location loc = (*lhs)->location();
    lhs = std::make_unique<BinaryOpNode>(loc, opTok.value(), std::move(*lhs), std::move(*rhs));
  }
  return lhs;
}

Result<std::unique_ptr<ASTNode>> Parser::parseAdditive() {
  auto lhs = parseMultiplicative();
  if (!lhs)
    return lhs;

  while (check(Token::Type::UNKNOWN) && (peek().value() == "+" || peek().value() == "-")) {
    Token opTok = advance();
    auto rhs = parseMultiplicative();
    if (!rhs)
      return rhs;

    Location loc = (*lhs)->location();
    lhs = std::make_unique<BinaryOpNode>(loc, opTok.value(), std::move(*lhs), std::move(*rhs));
  }
  return lhs;
}

Result<std::unique_ptr<ASTNode>> Parser::parseMultiplicative() {
  auto lhs = parsePrimary();
  if (!lhs)
    return lhs;

  while (check(Token::Type::UNKNOWN) && (peek().value() == "*" || peek().value() == "/")) {
    Token opTok = advance();
    auto rhs = parsePrimary();
    if (!rhs)
      return rhs;

    Location loc = (*lhs)->location();
    lhs = std::make_unique<BinaryOpNode>(loc, opTok.value(), std::move(*lhs), std::move(*rhs));
  }
  return lhs;
}

Result<std::unique_ptr<ASTNode>> Parser::parsePrimary() {
  spdlog::debug("Parsing primary expression");

  if (match(Token::Type::L_PAREN)) {
    auto expr = parseExpression();
    if (!expr)
      return expr;
    if (auto rparen = consume(Token::Type::R_PAREN, "Expected ')' after expression"); !rparen)
      return std::unexpected(rparen.error());
    return expr;
  }

  if (check(Token::Type::IDENTIFIER)) {
    const auto &identTok = advance();
    if (check(Token::Type::L_PAREN)) {
      return parseCall(identTok);
    }
    return parseVariable(identTok);
  }

  if (check(Token::Type::STRING))
    return parseString();
  if (check(Token::Type::NUMBER))
    return parseNumber();
  if (check(Token::Type::L_BRACKET))
    return parseKernel();

  return std::unexpected(
      CompileError{std::format("Unexpected token '{}' in expression", peek().value()), peek().location()});
}

Result<std::unique_ptr<ASTNode>> Parser::parseVariable(const Token &identifier) {
  spdlog::debug("Parsing variable");

  Token varTok = identifier;
  if (identifier.type() != Token::Type::IDENTIFIER) {
    auto consumed = consume(Token::Type::IDENTIFIER, "Expected identifier for variable");
    if (!consumed)
      return std::unexpected(consumed.error());
    varTok = *consumed;
  }

  return std::make_unique<VariableNode>(varTok.location(), varTok.value(), std::nullopt);
}

Result<std::unique_ptr<ASTNode>> Parser::parseKernel() {
  auto lbracket = consume(Token::Type::L_BRACKET, "Expected '[' to open kernel");
  if (!lbracket)
    return std::unexpected(lbracket.error());

  auto kernelNode = std::make_unique<KernelNode>(lbracket->location());

  while (!check(Token::Type::R_BRACKET) && !isAtEnd()) {
    if (auto innerL = consume(Token::Type::L_BRACKET, "Expected '[' for kernel row"); !innerL) {
      return std::unexpected(innerL.error());
    }

    std::vector<double> row;
    while (!check(Token::Type::R_BRACKET) && !isAtEnd()) {
      auto numTok = consume(Token::Type::NUMBER, "Expected number in kernel row");
      if (!numTok)
        return std::unexpected(numTok.error());

      row.push_back(std::stod(numTok->value()));

      if (!match(Token::Type::COMMA))
        break;
    }

    if (auto innerR = consume(Token::Type::R_BRACKET, "Expected ']' after kernel row"); !innerR) {
      return std::unexpected(innerR.error());
    }

    kernelNode->addRow(std::move(row));

    if (!match(Token::Type::COMMA))
      break;
  }

  if (auto rbracket = consume(Token::Type::R_BRACKET, "Expected ']' to close kernel"); !rbracket) {
    return std::unexpected(rbracket.error());
  }

  // Validate matrix dimensions
  if (!kernelNode->rows().empty()) {
    size_t cols = kernelNode->rows()[0].size();
    for (const auto &row : kernelNode->rows()) {
      if (row.size() != cols) {
        return std::unexpected(CompileError{"All rows in kernel must have same length"});
      }
    }
  }

  return kernelNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseString() {
  spdlog::debug("Parsing string");

  auto strTok = consume(Token::Type::STRING, "Expected string literal");
  if (!strTok)
    return std::unexpected(strTok.error());

  return std::make_unique<StringNode>(strTok->location(), utils::expandTilde(strTok->value()));
}

Result<std::unique_ptr<ASTNode>> Parser::parseNumber() {
  spdlog::debug("Parsing number");

  auto numTok = consume(Token::Type::NUMBER, "Expected double literal");
  if (!numTok)
    return std::unexpected(numTok.error());

  try {
    return std::make_unique<NumberNode>(numTok->location(), std::stod(numTok->value()));
  } catch (const std::invalid_argument &) {
    return std::unexpected(
        CompileError{std::format("Invalid double literal '{}'", numTok->value()), numTok->location()});
  } catch (const std::out_of_range &) {
    return std::unexpected(
        CompileError{std::format("Double literal '{}' out of range", numTok->value()), numTok->location()});
  }
}