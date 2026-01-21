#include "parser.h"

#include <iostream>
#include <format>

#include "spdlog/spdlog.h"
#include "error.h"

using namespace picceler;

Parser::Parser() : _lexer() {}

void Parser::setSource(const std::string &source) { _lexer.setSource(source); }

Result<std::vector<Token>> Parser::getTokens() {
  auto tokens = _lexer.tokenizeAll();
  if (!tokens) {
    return std::unexpected(CompileError{"Couldn't retrieve all tokens from the lexer"});
  }
  return tokens.value();
}

Result<std::unique_ptr<ModuleNode>> Parser::parse() {
  spdlog::info("Starting parsing process");
  auto module = std::make_unique<ModuleNode>();
  while (true) {
    auto stmt = parseStatement();
    if (!stmt) {
      spdlog::error("{}", stmt.error().message());
      return std::unexpected(CompileError{"Parsing process failed"});
    }
    auto statement = std::move(stmt.value());
    if (!statement) {
      // Reached EOF: parseStatement returned a successful nullptr to signal end
      break;
    }
    spdlog::info("Parsed statement: {}", statement->toString());
    module->statements.push_back(std::move(statement));
  }
  spdlog::info("Finished parsing process");
  return module;
}

Result<std::unique_ptr<ASTNode>> Parser::parseStatement() {
  auto tokenResult = _lexer.peekToken();
  if (!tokenResult) {
    spdlog::error("{}", tokenResult.error().message());
    return std::unexpected(CompileError{"parseStatement unexpected stop of parsing"});
  }
  auto token = *tokenResult;
  if (token._type == Token::Type::IDENTIFIER) {
    spdlog::debug("Parsing statement starting with identifier '{}'", token._value);
    auto identifierResult = _lexer.nextToken();
    if (!identifierResult) {
      spdlog::error("{}", identifierResult.error().message());
      return std::unexpected(CompileError{"Failed to consume identifier token"});
    }
    auto identifier = *identifierResult;
    auto nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token"});
    }
    auto nextToken = *nextTokenResult;
    if (nextToken._type == Token::Type::SYMBOL) {
      if (nextToken._value == "=") {
        return parseAssignment(identifier);
      } else if (nextToken._value == "(") {
        return parseCall(identifier);
      }
    }
    return std::unexpected(CompileError{
        std::format("Unexpected token '{}' at {}:{}", nextToken._value, nextToken._line, nextToken._column)});
  } else if (token._type == Token::Type::EOF_TOKEN) {
    return nullptr;
  } else {
    return std::unexpected(
        CompileError{std::format("Unexpected token '{}' at {}:{}", token._value, token._line, token._column)});
  }
}

Result<std::unique_ptr<ASTNode>> Parser::parseAssignment(Token identifier) {
  spdlog::debug("Parsing assignment for identifier '{}'", identifier._value);
  auto eqTokenResult = _lexer.nextToken(); // consume '='
  if (!eqTokenResult) {
    spdlog::error("{}", eqTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '=' token in assignment"});
  }
  auto eqToken = *eqTokenResult;
  if (eqToken._type != Token::Type::SYMBOL || eqToken._value != "=") {
    return std::unexpected(
        CompileError{std::format("Expected '=' after identifier at {}:{}", eqToken._line, eqToken._column)});
  }
  auto exprResult = parseExpression();
  if (!exprResult) {
    spdlog::error("{}", exprResult.error().message());
    return std::unexpected(CompileError{"Failed to parse expression in assignment"});
  }
  auto assignNode = std::make_unique<AssignmentNode>();
  auto leftVarResult = parseVariable(identifier);
  if (!leftVarResult) {
    spdlog::error("{}", leftVarResult.error().message());
    return std::unexpected(CompileError{"Failed to parse variable in assignment"});
  }
  auto leftVar = std::move(leftVarResult.value());
  assignNode->lhs = std::unique_ptr<VariableNode>(dynamic_cast<VariableNode *>(leftVar.release()));
  assignNode->rhs = std::move(exprResult.value());
  return assignNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseCall(Token identifier) {
  spdlog::debug("Parsing function call for identifier '{}'", identifier._value);
  auto lparenTokenResult = _lexer.nextToken(); // consume '('
  if (!lparenTokenResult) {
    spdlog::error("{}", lparenTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '(' token in function call"});
  }
  auto lparenToken = *lparenTokenResult;
  if (lparenToken._type != Token::Type::SYMBOL || lparenToken._value != "(") {
    return std::unexpected(
        CompileError{std::format("Expected '(' after function name at {}:{}", lparenToken._line, lparenToken._column)});
  }
  auto callNode = std::make_unique<CallNode>();
  callNode->callee = identifier._value;

  while (true) {
    auto nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token in function call"});
    }
    auto nextToken = *nextTokenResult;
    if (nextToken._type == Token::Type::SYMBOL && nextToken._value == ")") {
      auto rparenTokenResult = _lexer.nextToken(); // consume ')'
      if (!rparenTokenResult) {
        spdlog::error("{}", rparenTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume ')' token in function call"});
      }
      break;
    }
    auto argResult = parseExpression();
    if (!argResult) {
      spdlog::error("{}", argResult.error().message());
      return std::unexpected(CompileError{"Failed to parse argument in function call"});
    }
    auto arg = std::move(argResult.value());
    callNode->arguments.push_back(std::move(arg));

    nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token in function call"});
    }
    nextToken = *nextTokenResult;
    if (nextToken._type == Token::Type::SYMBOL && nextToken._value == ",") {
      auto commaTokenResult = _lexer.nextToken(); // consume ','
      if (!commaTokenResult) {
        spdlog::error("{}", commaTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume ',' token in function call"});
      }
    } else if (nextToken._type == Token::Type::SYMBOL && nextToken._value == ")") {
      continue;
    } else {
      return std::unexpected(CompileError{
          std::format("Expected ',' or ')' in function call at {}:{}", nextToken._line, nextToken._column)});
    }
  }

  return callNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseExpression() {
  spdlog::debug("Parsing expression");
  auto tokenResult = _lexer.peekToken();
  if (!tokenResult) {
    spdlog::error("{}", tokenResult.error().message());
    return std::unexpected(CompileError{"parseExpression unexpected stop of parsing"});
  }
  auto token = *tokenResult;
  if (token._type == Token::Type::IDENTIFIER) {
    auto currentTokenResult = _lexer.nextToken();
    if (!currentTokenResult) {
      spdlog::error("{}", currentTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume identifier token in expression"});
    }
    auto currentToken = *currentTokenResult;
    auto nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token in expression"});
    }
    auto nextToken = *nextTokenResult;
    if (nextToken._type == Token::Type::SYMBOL && nextToken._value == "(") {
      return parseCall(currentToken);
    } else {
      return parseVariable(currentToken);
    }
  } else if (token._type == Token::Type::STRING) {
    return parseString();
  } else if (token._type == Token::Type::NUMBER) {
    return parseNumber();
  } else if (token._type == Token::Type::SYMBOL && token._value == "[") {
    return parseKernel();
  } else {
    return std::unexpected(
        CompileError{std::format("Unexpected token '{}' at {}:{}", token._value, token._line, token._column)});
  }
}

Result<std::unique_ptr<ASTNode>> Parser::parseVariable(Token identifier) {
  spdlog::debug("Parsing variable");
  // If the passed token is not an identifier, consume the next token
  if (identifier._type != Token::Type::IDENTIFIER) {
    auto identifierResult = _lexer.nextToken();
    if (!identifierResult) {
      spdlog::error("{}", identifierResult.error().message());
      return std::unexpected(CompileError{"Failed to consume identifier token in variable"});
    }
    identifier = *identifierResult;
  }

  auto varNode = std::make_unique<VariableNode>();
  varNode->name = identifier._value;
  return varNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseKernel() {
  spdlog::debug("Parsing kernel");
  auto lbracketTokenResult = _lexer.nextToken(); // consume '['
  if (!lbracketTokenResult) {
    spdlog::error("{}", lbracketTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '[' token in kernel"});
  }
  auto lbracketToken = *lbracketTokenResult;
  if (lbracketToken._type != Token::Type::SYMBOL || lbracketToken._value != "[") {
    return std::unexpected(
        CompileError{std::format("Expected '[' at {}:{}", lbracketToken._line, lbracketToken._column)});
  }

  auto kernelNode = std::make_unique<KernelNode>();

  while (true) {
    auto peekedtokenResult = _lexer.peekToken();
    if (!peekedtokenResult) {
      spdlog::error("{}", peekedtokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek token in kernel"});
    }
    auto peekToken = *peekedtokenResult;

    // End of kernel
    if (peekToken._type == Token::Type::SYMBOL && peekToken._value == "]") {
      auto rbracket = _lexer.nextToken(); // consume ']'
      if (!rbracket) {
        spdlog::error("{}", rbracket.error().message());
        return std::unexpected(CompileError{"Failed to consume ']' token after kernel"});
      }
      break;
    }

    // Expect a row: '[' NUMBER (',' NUMBER)* ']'

    if (!(peekToken._type == Token::Type::SYMBOL && peekToken._value == "[")) {
      return std::unexpected(
          CompileError{std::format("Expected '[' for kernel row at {}:{}", peekToken._line, peekToken._column)});
    }

    // consume inner '['
    auto innerLBracketTokenResult = _lexer.nextToken();
    if (!innerLBracketTokenResult) {
      spdlog::error("{}", innerLBracketTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume inner '[' token in kernel"});
    }

    std::vector<double> row;

    while (true) {
      auto innerTokenResult = _lexer.peekToken();
      if (!innerTokenResult) {
        spdlog::error("{}", innerTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to peek token in kernel"});
      }
      auto innerToken = *innerTokenResult;

      if (innerToken._type == Token::Type::SYMBOL && innerToken._value == "]") {
        // empty row allowed? treat as end of row
        auto consumed = _lexer.nextToken(); // consume inner ']'
        if (!consumed) {
          spdlog::error("{}", consumed.error().message());
          return std::unexpected(CompileError{"Failed to consume inner ']' token in kernel"});
        }
        break;
      }

      // expect a number
      if (innerToken._type != Token::Type::NUMBER) {
        return std::unexpected(
            CompileError{std::format("Expected number in kernel at {}:{}", innerToken._line, innerToken._column)});
      }
      // consume number
      auto numTokenResult = _lexer.nextToken();
      if (!numTokenResult) {
        spdlog::error("{}", numTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume number token in kernel"});
      }
      auto numToken = *numTokenResult;
      row.push_back(std::stod(numToken._value));

      // after a number, expect ',' or ']'
      auto afterNumResult = _lexer.peekToken();
      if (!afterNumResult) {
        spdlog::error("{}", afterNumResult.error().message());
        return std::unexpected(CompileError{"Failed to peek token in kernel"});
      }
      auto afterNum = *afterNumResult;
      if (afterNum._type == Token::Type::SYMBOL && afterNum._value == ",") {
        auto commaTokenResult = _lexer.nextToken(); // consume comma
        if (!commaTokenResult) {
          spdlog::error("{}", commaTokenResult.error().message());
          return std::unexpected(CompileError{"Failed to consume comma token in kernel"});
        }
        continue; // continue reading numbers for this row
      } else if (afterNum._type == Token::Type::SYMBOL && afterNum._value == "]") {
        // consume closing inner ']'
        auto consumed = _lexer.nextToken();
        if (!consumed) {
          spdlog::error("{}", consumed.error().message());
          return std::unexpected(CompileError{"Failed to consume inner ']' token in kernel"});
        }
        break;
      } else {
        return std::unexpected(
            CompileError{std::format("Expected ',' or ']' in kernel row at {}:{}", afterNum._line, afterNum._column)});
      }
    }

    // push parsed row
    kernelNode->rows.push_back(row);

    // after a row, look for either ',' (another row) or ']' (end of kernel)
    auto nextResult = _lexer.peekToken();
    if (!nextResult) {
      spdlog::error("{}", nextResult.error().message());
      return std::unexpected(CompileError{"Failed to peek token in kernel"});
    }
    auto next = *nextResult;
    if (next._type == Token::Type::SYMBOL && next._value == ",") {
      auto commaTokenResult = _lexer.nextToken(); // consume comma between rows
      if (!commaTokenResult) {
        spdlog::error("{}", commaTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume comma token in kernel"});
      }
      // continue to next row
    } else if (next._type == Token::Type::SYMBOL && next._value == "]") {
      // consume closing ']'
      auto r = _lexer.nextToken();
      if (!r) {
        spdlog::error("{}", r.error().message());
        return std::unexpected(CompileError{"Failed to consume ']' token after kernel rows"});
      }
      break;
    } else {
      return std::unexpected(
          CompileError{std::format("Expected ',' or ']' after kernel row at {}:{}", next._line, next._column)});
    }
  }

  // Validate dimensions: only 2D kernels supported; ensure all rows same length
  if (!kernelNode->rows.empty()) {
    size_t cols = kernelNode->rows[0].size();
    for (const auto &row : kernelNode->rows) {
      if (row.size() != cols) {
        return std::unexpected(CompileError{"All rows in kernel must have same length"});
      }
    }
  }

  return kernelNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseString() {
  spdlog::debug("Parsing string");
  auto tokenResult = _lexer.nextToken(); // consume string
  if (!tokenResult) {
    spdlog::error("{}", tokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume string token"});
  }
  auto token = *tokenResult;
  if (token._type != Token::Type::STRING) {
    return std::unexpected(CompileError{std::format("Expected string at {}:{}", token._line, token._column)});
  }
  auto strNode = std::make_unique<StringNode>();
  strNode->value = token._value;
  return strNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseNumber() {
  spdlog::debug("Parsing number");
  auto tokenResult = _lexer.nextToken(); // consume number
  if (!tokenResult) {
    spdlog::error("{}", tokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume number token"});
  }
  auto token = *tokenResult;
  if (token._type != Token::Type::NUMBER) {
    return std::unexpected(CompileError{std::format("Expected number at {}:{}", token._line, token._column)});
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
