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
   * @brief Constructs a Parser.
   */
  Parser();

  /**
   * @brief Sets the source string for the parser.
   * @param source The source string to read from.
   */
  void setSourceString(std::string_view source);

  /**
   * @brief Sets the source file for the parser.
   * @param source The source file to read from.
   */
  Result<void> setSource(const std::string &filepath);

  /**
   * @brief Retrieves all tokens from the lexer.
   * @return A vector of all tokens.
   */
  Result<std::vector<Token>> getTokens();

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
  /**
   * \name Parse helper functions
   * \{
   */
  Result<std::unique_ptr<ASTNode>> parseStatement();
  Result<std::unique_ptr<ASTNode>> parseExpression();
  Result<std::unique_ptr<ASTNode>> parseFunctionDefinition(const Token &defToken);
  Result<std::unique_ptr<ASTNode>> parseAssignment(const Token &identifier);
  Result<std::unique_ptr<ASTNode>> parseCall(const Token &identifier);
  Result<std::unique_ptr<ASTNode>> parseVariable(const Token &identifier = Token{Token::Type::UNKNOWN, "", {}});
  Result<std::unique_ptr<ASTNode>> parseKernel();
  Result<std::unique_ptr<ASTNode>> parseString();
  Result<std::unique_ptr<ASTNode>> parseNumber();
  /**
   * \}
   */

private:
  Lexer _lexer;
};

} // namespace picceler