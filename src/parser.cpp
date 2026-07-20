#include "parser.h"

#include <format>
#include <stdexcept>

#include "spdlog/spdlog.h"
#include "error.h"
#include "utils.h"

using namespace picceler;

Parser::Parser() : _lexer() {}

Result<void> Parser::setSource(const std::string &source) { return _lexer.setSource(source); }
void Parser::setSourceString(std::string_view source) { _lexer.setSourceString(source); }

Result<std::vector<Token>> Parser::getTokens() {
  auto tokens = _lexer.tokenizeAll();
  if (!tokens) {
    return std::unexpected(CompileError{"Couldn't retrieve all tokens from the lexer"});
  }
  return tokens.value();
}

Result<std::unique_ptr<ModuleNode>> Parser::parse() {
  spdlog::info("Starting parsing process");
  auto module = std::make_unique<ModuleNode>(Location(0, 0));
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
    module->addStatement(std::move(statement));
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
  const auto &token = *tokenResult;
  if (token == Token::Type::IDENTIFIER) {
    spdlog::debug("Parsing statement starting with identifier '{}'", token.value());
    auto identifierResult = _lexer.nextToken();
    if (!identifierResult) {
      spdlog::error("{}", identifierResult.error().message());
      return std::unexpected(CompileError{"Failed to consume identifier token"});
    }
    const auto &identifier = *identifierResult;
    auto nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token"});
    }
    const auto &nextToken = *nextTokenResult;
    if (nextToken == Token::Type::ASSIGN) {
      return parseAssignment(identifier);
    } else if (nextToken == Token::Type::L_PAREN) {
      return parseCall(identifier);
    }
    return std::unexpected(
        CompileError{std::format("Unexpected token '{}' after identifier", nextToken.value()), nextToken.location()});

  } else if (token == Token::Type::KW_DEF) {
    auto defTokenResult = _lexer.nextToken(); // consume 'def'
    if (!defTokenResult) {
      spdlog::error("{}", defTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume 'def' token"});
    }
    auto defToken = std::move(*defTokenResult);
    return parseFunctionDefinition(defToken);
  } else if (token == Token::Type::EOF_TOKEN) {
    return nullptr;
  } else {
    return std::unexpected(CompileError{std::format("Unexpected token '{}'", token.value()), token.location()});
  }
}

Result<std::unique_ptr<ASTNode>> Parser::parseFunctionDefinition([[maybe_unused]] const Token &defToken) {
  const auto &nameTokenResult = _lexer.nextToken(); // consume function name
  if (!nameTokenResult) {
    spdlog::error("{}", nameTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume function name token"});
  }
  if (nameTokenResult->type() != Token::Type::IDENTIFIER) {
    return std::unexpected(CompileError{"Expected function name after 'def'", nameTokenResult->location()});
  }
  auto funcName = nameTokenResult->value();
  auto funcNode = std::make_unique<FunctionNode>(nameTokenResult->location(), funcName);
  const auto &lParenTokenResult = _lexer.nextToken(); // consume '('
  if (!lParenTokenResult) {
    spdlog::error("{}", lParenTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '(' token in function definition"});
  }
  const auto &lParenToken = *lParenTokenResult;
  if (lParenToken.type() != Token::Type::L_PAREN) {
    return std::unexpected(CompileError{"Expected '(' in function definition"});
  }
  auto tokenItter = _lexer.peekToken();
  if (!tokenItter) {
    spdlog::error("{}", tokenItter.error().message());
    return std::unexpected(CompileError{"Failed to peek token in function definition"});
  }
  while (tokenItter.value() != Token::Type::R_PAREN) {
    auto paramTokenResult = _lexer.nextToken(); // consume parameter -- expected name
    if (!paramTokenResult) {
      spdlog::error("{}", paramTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume parameter token in function definition"});
    }
    const auto &paramToken = *paramTokenResult;
    if (paramToken != Token::Type::IDENTIFIER) {
      return std::unexpected(CompileError{"Expected parameter name", paramToken.location()});
    }
    auto colonTokenResult = _lexer.nextToken(); // consume ':'
    if (!colonTokenResult) {
      spdlog::error("{}", colonTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume ':' token in function definition"});
    }
    const auto &colonToken = *colonTokenResult;
    if (colonToken != Token::Type::COLON) {
      return std::unexpected(CompileError{std::format("Expected ':' after parameter name"), colonToken.location()});
    }
    auto typeTokenResult = _lexer.nextToken(); // consume type
    if (!typeTokenResult) {
      spdlog::error("{}", typeTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume type token in function definition"});
    }
    const auto &typeToken = *typeTokenResult;
    if (typeToken != Token::Type::TYPE) {
      return std::unexpected(CompileError{"Expected type after ':'", typeToken.location()});
    }
    funcNode->addParameter(paramToken.value(), typeToken.value());
    tokenItter = _lexer.peekToken();
    if (!tokenItter) {
      spdlog::error("{}", tokenItter.error().message());
      return std::unexpected(CompileError{"Failed to peek token in function definition"});
    }
    if (tokenItter.value() == Token::Type::COMMA) {
      auto commaTokenResult = _lexer.nextToken(); // consume ','
      if (!commaTokenResult) {
        spdlog::error("{}", commaTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume ',' token in function definition"});
      }
    }
    tokenItter = _lexer.peekToken(); // could be ')' or another parameter
    if (!tokenItter) {
      spdlog::error("{}", tokenItter.error().message());
      return std::unexpected(CompileError{"Failed to consume token in function definition"});
    }
  }
  auto rParenTokenResult = _lexer.nextToken(); // consume ')'
  if (!rParenTokenResult) {
    spdlog::error("{}", rParenTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume ')' token in function definition"});
  }
  const auto &rParenToken = *rParenTokenResult;
  if (rParenToken != Token::Type::R_PAREN) {
    return std::unexpected(CompileError{"Expected ')' after parameters", rParenToken.location()});
  }

  auto lBraceTokenResult = _lexer.nextToken(); // consume '{'
  if (!lBraceTokenResult) {
    spdlog::error("{}", lBraceTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '{' token in function definition"});
  }
  const auto &lBraceToken = *lBraceTokenResult;
  if (lBraceToken != Token::Type::L_BRACE) {
    return std::unexpected(CompileError{std::format("Expected '{{' after function signature"), lBraceToken.location()});
  }

  tokenItter = _lexer.peekToken();
  if (!tokenItter) {
    spdlog::error("{}", tokenItter.error().message());
    return std::unexpected(CompileError{"Failed to peek token in function definition"});
  }
  while (tokenItter.value() != Token::Type::R_BRACE) {
    auto stmtResult = parseStatement();
    if (!stmtResult) {
      spdlog::error("{}", stmtResult.error().message());
      return std::unexpected(CompileError{"Failed to parse statement in function body"});
    }
    auto stmt = std::move(stmtResult.value());
    if (!stmt) {
      return std::unexpected(CompileError{"Unexpected end of function body"});
    }
    funcNode->addBodyStatement(std::move(stmt));
    tokenItter = _lexer.peekToken();
    if (!tokenItter) {
      spdlog::error("{}", tokenItter.error().message());
      return std::unexpected(CompileError{"Failed to peek token in function definition"});
    }
  }

  auto rBraceTokenResult = _lexer.nextToken(); // consume '}'
  if (!rBraceTokenResult) {
    spdlog::error("{}", rBraceTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '}' token in function definition"});
  }
  const auto &rBraceToken = *rBraceTokenResult;
  if (rBraceToken != Token::Type::R_BRACE) {
    return std::unexpected(CompileError{std::format("Expected '}}' after function body"), rBraceToken.location()});
  }

  return funcNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseAssignment(const Token &identifier) {
  spdlog::debug("Parsing assignment for identifier '{}'", identifier.value());
  auto eqTokenResult = _lexer.nextToken(); // consume '='
  if (!eqTokenResult) {
    spdlog::error("{}", eqTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '=' token in assignment"});
  }
  const auto &eqToken = *eqTokenResult;
  if (eqToken != Token::Type::ASSIGN) {
    return std::unexpected(CompileError{"Expected '=' after identifier", eqToken.location()});
  }
  auto exprResult = parseExpression();
  if (!exprResult) {
    spdlog::error("{}", exprResult.error().message());
    return std::unexpected(CompileError{"Failed to parse expression in assignment"});
  }
  auto exprVar = std::move(exprResult.value());
  auto leftVarResult = parseVariable(identifier);
  if (!leftVarResult) {
    spdlog::error("{}", leftVarResult.error().message());
    return std::unexpected(CompileError{"Failed to parse variable in assignment"});
  }
  auto leftVar = std::move(leftVarResult.value());
  if (dynamic_cast<VariableNode *>(leftVar.get()) == nullptr) {
    return std::unexpected(CompileError{"Left-hand side of assignment must be a variable"});
  }
  auto leftVarNode = std::unique_ptr<VariableNode>(static_cast<VariableNode *>(leftVar.release()));
  auto assignNode =
      std::make_unique<AssignmentNode>(leftVarNode->location(), std::move(leftVarNode), std::move(exprVar));
  return assignNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseCall(const Token &identifier) {
  spdlog::debug("Parsing function call for identifier '{}'", identifier.value());
  auto lparenTokenResult = _lexer.nextToken(); // consume '('
  if (!lparenTokenResult) {
    spdlog::error("{}", lparenTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '(' token in function call"});
  }
  const auto &lparenToken = *lparenTokenResult;
  if (lparenToken != Token::Type::L_PAREN) {
    return std::unexpected(CompileError{std::format("Expected '(' after function name"), lparenToken.location()});
  }
  auto callNode = std::make_unique<CallNode>(identifier.location(), identifier.value());

  while (true) {
    auto nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token in function call"});
    }
    auto nextToken = *nextTokenResult;
    if (nextToken == Token::Type::R_PAREN) {
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
    callNode->addArgument(std::move(arg));

    nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token in function call"});
    }
    nextToken = *nextTokenResult;
    if (nextToken == Token::Type::COMMA) {
      auto commaTokenResult = _lexer.nextToken(); // consume ','
      if (!commaTokenResult) {
        spdlog::error("{}", commaTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume ',' token in function call"});
      }
    } else if (nextToken == Token::Type::R_PAREN) {
      continue;
    } else {
      return std::unexpected(CompileError{std::format("Expected ',' or ')' in function call "), nextToken.location()});
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
  const auto &token = *tokenResult;
  if (token == Token::Type::IDENTIFIER) {
    auto currentTokenResult = _lexer.nextToken();
    if (!currentTokenResult) {
      spdlog::error("{}", currentTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to consume identifier token in expression"});
    }
    const auto &currentToken = *currentTokenResult;
    auto nextTokenResult = _lexer.peekToken();
    if (!nextTokenResult) {
      spdlog::error("{}", nextTokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek next token in expression"});
    }
    const auto &nextToken = *nextTokenResult;
    if (nextToken == Token::Type::L_PAREN) {
      return parseCall(currentToken);
    } else {
      return parseVariable(currentToken);
    }
  } else if (token == Token::Type::STRING) {
    return parseString();
  } else if (token == Token::Type::NUMBER) {
    return parseNumber();
  } else if (token == Token::Type::L_BRACKET) {
    return parseKernel();
  } else {
    return std::unexpected(CompileError{std::format("Unexpected token '{}'", token.value()), token.location()});
  }
}

Result<std::unique_ptr<ASTNode>> Parser::parseVariable(const Token &identifier) {
  spdlog::debug("Parsing variable");
  std::string varName = identifier.value();
  Location loc = identifier.location();
  // If the passed token is not an identifier, consume the next token
  if (identifier != Token::Type::IDENTIFIER) {
    auto identifierResult = _lexer.nextToken();
    if (!identifierResult) {
      spdlog::error("{}", identifierResult.error().message());
      return std::unexpected(CompileError{"Failed to consume identifier token in variable"});
    }
    varName = identifierResult->value();
    loc = identifierResult->location();
  }

  auto varNode = std::make_unique<VariableNode>(loc, varName, std::nullopt);
  return varNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseKernel() {
  spdlog::debug("Parsing kernel");
  auto lbracketTokenResult = _lexer.nextToken(); // consume '['
  if (!lbracketTokenResult) {
    spdlog::error("{}", lbracketTokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume '[' token in kernel"});
  }
  const auto &lbracketToken = *lbracketTokenResult;
  if (lbracketToken != Token::Type::L_BRACKET) {
    return std::unexpected(CompileError{"Expected '['", lbracketToken.location()});
  }

  auto kernelNode = std::make_unique<KernelNode>(lbracketToken.location());

  while (true) {
    auto peekedtokenResult = _lexer.peekToken();
    if (!peekedtokenResult) {
      spdlog::error("{}", peekedtokenResult.error().message());
      return std::unexpected(CompileError{"Failed to peek token in kernel"});
    }
    const auto &peekToken = *peekedtokenResult;

    // End of kernel
    if (peekToken == Token::Type::R_BRACKET) {
      auto rbracket = _lexer.nextToken(); // consume ']'
      if (!rbracket) {
        spdlog::error("{}", rbracket.error().message());
        return std::unexpected(CompileError{"Failed to consume ']' token after kernel"});
      }
      break;
    }

    // Expect a row: '[' NUMBER (',' NUMBER)* ']'

    if (!(peekToken == Token::Type::L_BRACKET)) {
      return std::unexpected(CompileError{"Expected '[' for kernel row", peekToken.location()});
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
      const auto &innerToken = *innerTokenResult;

      if (innerToken == Token::Type::R_BRACKET) {
        // empty row allowed? treat as end of row
        auto consumed = _lexer.nextToken(); // consume inner ']'
        if (!consumed) {
          spdlog::error("{}", consumed.error().message());
          return std::unexpected(CompileError{"Failed to consume inner ']' token in kernel"});
        }
        break;
      }

      // expect a number
      if (innerToken != Token::Type::NUMBER) {
        return std::unexpected(CompileError{"Expected number in kernel", innerToken.location()});
      }
      // consume number
      auto numTokenResult = _lexer.nextToken();
      if (!numTokenResult) {
        spdlog::error("{}", numTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume number token in kernel"});
      }
      const auto &numToken = *numTokenResult;
      row.push_back(std::stod(numToken.value()));

      // after a number, expect ',' or ']'
      auto afterNumResult = _lexer.peekToken();
      if (!afterNumResult) {
        spdlog::error("{}", afterNumResult.error().message());
        return std::unexpected(CompileError{"Failed to peek token in kernel"});
      }
      const auto &afterNum = *afterNumResult;
      if (afterNum == Token::Type::COMMA) {
        auto commaTokenResult = _lexer.nextToken(); // consume comma
        if (!commaTokenResult) {
          spdlog::error("{}", commaTokenResult.error().message());
          return std::unexpected(CompileError{"Failed to consume comma token in kernel"});
        }
        continue; // continue reading numbers for this row
      } else if (afterNum == Token::Type::R_BRACKET) {
        // consume closing inner ']'
        auto consumed = _lexer.nextToken();
        if (!consumed) {
          spdlog::error("{}", consumed.error().message());
          return std::unexpected(CompileError{"Failed to consume inner ']' token in kernel"});
        }
        break;
      } else {
        return std::unexpected(CompileError{"Expected ',' or ']' in kernel row", afterNum.location()});
      }
    }

    // push parsed row
    kernelNode->addRow(row);

    // after a row, look for either ',' (another row) or ']' (end of kernel)
    auto nextResult = _lexer.peekToken();
    if (!nextResult) {
      spdlog::error("{}", nextResult.error().message());
      return std::unexpected(CompileError{"Failed to peek token in kernel"});
    }
    const auto &next = *nextResult;
    if (next == Token::Type::COMMA) {
      auto commaTokenResult = _lexer.nextToken(); // consume comma between rows
      if (!commaTokenResult) {
        spdlog::error("{}", commaTokenResult.error().message());
        return std::unexpected(CompileError{"Failed to consume comma token in kernel"});
      }
      // continue to next row
    } else if (next == Token::Type::R_BRACKET) {
      // consume closing ']'
      auto r = _lexer.nextToken();
      if (!r) {
        spdlog::error("{}", r.error().message());
        return std::unexpected(CompileError{"Failed to consume ']' token after kernel rows"});
      }
      break;
    } else {
      return std::unexpected(CompileError{"Expected ',' or ']' after kernel row", next.location()});
    }
  }

  // Validate dimensions: only 2D kernels supported; ensure all rows same length
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
  auto tokenResult = _lexer.nextToken(); // consume string
  if (!tokenResult) {
    spdlog::error("{}", tokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume string token"});
  }
  const auto &token = *tokenResult;
  if (token != Token::Type::STRING) {
    return std::unexpected(CompileError{"Expected string", token.location()});
  }
  // TODO: Implement a compiler flag to let the user not expand tilde if it is not needed
  auto strNode = std::make_unique<StringNode>(token.location(), utils::expandTilde(token.value()));
  return strNode;
}

Result<std::unique_ptr<ASTNode>> Parser::parseNumber() {
  spdlog::debug("Parsing number");
  auto tokenResult = _lexer.nextToken(); // consume number
  if (!tokenResult) {
    spdlog::error("{}", tokenResult.error().message());
    return std::unexpected(CompileError{"Failed to consume number token"});
  }
  const auto &token = *tokenResult;
  if (token != Token::Type::NUMBER) {
    return std::unexpected(CompileError{"Expected number", token.location()});
  }
  std::unique_ptr<NumberNode> numNode = nullptr;
  try {
    numNode = std::make_unique<NumberNode>(token.location(), std::stod(token.value()));
  } catch (const std::invalid_argument &) {
    return std::unexpected(CompileError{std::format("Invalid double literal '{}'", token.value()), token.location()});
  } catch (const std::out_of_range &) {
    return std::unexpected(
        CompileError{std::format("Double literal '{}' out of range", token.value()), token.location()});
  }
  return numNode;
}

void Parser::printAST(const std::unique_ptr<ModuleNode> &node, int indent) {
  if (!node)
    return;
  spdlog::info("{}", node->toString());
  for (const auto &stmt : node->statements()) {
    spdlog::info("{}", stmt->toString());
  }
}
