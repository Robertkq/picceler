#pragma once

#include <memory>
#include <vector>
#include <string>

#include "ast.h"
#include "lexer.h"

namespace picceler {

/**
 * @brief The Parser class that converts tokens into an AST.
 */
class Parser {
public:
  /**
   * @brief Constructs a Parser without tokens.
   */
  Parser();

  /**
   * @brief Constructs a Parser with a given vector of tokens.
   */
  explicit Parser(std::vector<Token> tokens);

  void setTokens(std::vector<Token> tokens) {
    _tokens = std::move(tokens);
    _index = 0;
  }

  /**
   * @brief Parses the tokens into an AST.
   * @return The root of the AST.
   */
  Result<std::unique_ptr<ModuleNode>> parse();

  /**
   * @brief Prints the AST in a human-readable format.
   * @param node The root of the AST.
   * @param indent The current indentation level (used for formatting) (not
   * working currently).
   */
  void printAST(const std::unique_ptr<ModuleNode> &node, int indent = 0);

private:
  const Token &peek() const;
  const Token &peekAhead(std::size_t offset) const;
  bool check(Token::Type type) const;

  const Token &advance();

  bool match(Token::Type type);

  Result<Token> consume(Token::Type type, std::string_view errorMessage);

  bool isAtEnd() const;

  /**
   * \name Parse explicit functions for each type of statement or expression.
   * \{
   */
  Result<std::unique_ptr<ASTNode>> parseStatement();
  Result<std::unique_ptr<ASTNode>> parseExpression();
  Result<std::unique_ptr<ASTNode>> parseFunctionDefinition();
  Result<std::unique_ptr<ASTNode>> parseAssignment(const Token &identifier);
  Result<std::unique_ptr<ASTNode>> parseCall(const Token &identifier);
  Result<std::unique_ptr<ASTNode>> parseVariable(const Token &identifier = Token{Token::Type::UNKNOWN, "", {}});
  Result<std::unique_ptr<ASTNode>> parseKernel();
  Result<std::unique_ptr<ASTNode>> parseString();
  Result<std::unique_ptr<ASTNode>> parseNumber();

  Result<std::unique_ptr<ASTNode>> parseRelational();
  Result<std::unique_ptr<ASTNode>> parseAdditive();
  Result<std::unique_ptr<ASTNode>> parseMultiplicative();
  Result<std::unique_ptr<ASTNode>> parsePrimary();
  Result<std::unique_ptr<ASTNode>> parseIfStatement();
  /**
   * \}
   */

private:
  // Lexer _lexer;
  std::vector<Token> _tokens;
  std::size_t _index;
};

} // namespace picceler