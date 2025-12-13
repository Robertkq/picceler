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
   * @brief Sets the source file for the parser.
   * @param source The source file to read from.
   */
  void setSource(const std::string &source);

  /**
   * @brief Retrieves all tokens from the lexer.
   * @return A vector of all tokens.
   */
  std::vector<Token> getTokens();

  /**
   * @brief Parses the tokens into an AST.
   * @return The root of the AST.
   */
  std::unique_ptr<ModuleNode> parse();

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
  std::unique_ptr<ASTNode> parseStatement();
  std::unique_ptr<ASTNode> parseExpression();
  std::unique_ptr<ASTNode> parseAssignment(Token identifier);
  std::unique_ptr<ASTNode> parseCall(Token identifier);
  std::unique_ptr<ASTNode> parseVariable(Token identifier = Token{
                                             Token::Type::UNKNOWN, "", 0, 0});
  std::unique_ptr<ASTNode> parseString();
  std::unique_ptr<ASTNode> parseNumber();
  /**
   * \}
   */

private:
  Lexer _lexer;
};

} // namespace picceler