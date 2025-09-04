/**
 * @file lexer.h
 * @brief Lexer for the picceler programming language.
 *
 * The lexer tokenizes strings from a source file into tokens for parsing.
 *
 */

#pragma once

#include <spdlog/spdlog.h>

#include <cstdint>
#include <fstream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace picceler {

/**
 * @struct Token
 * @brief Represents a token produced by the lexer.
 */
struct Token {
  /** @brief The type of the token. */
  enum class Type : size_t {
    IDENTIFIER,
    NUMBER,
    STRING,
    SYMBOL,
    EOF_TOKEN,
    UNKNOWN
  };

  /**
   * @brief Converts the token type to a string representation.
   * @return The string representation of the token type.
   */
  std::string toString() const;

  Type _type;
  std::string _value;
  size_t _line;
  size_t _column;
};

/**
 * @class Lexer
 * @brief Lexical analyzer for the picceler programming language.
 */
class Lexer {
public:
  /**
   * @brief Constructs a Lexer
   */
  Lexer();

  /**
   * @brief Sets the source file for the lexer.
   * @param source The source file to read from.
   */
  void setSource(const std::string &source);

  /**
   * @brief Returns the next token from the input.
   * @return The next token.
   */
  Token nextToken();

  /**
   * @brief Returns the next token without advancing the input.
   * @return The next token.
   */
  Token peekToken();

  /**
   * @brief Skips whitespace characters in the input.
   */
  void skipWhitespace();

  /**
   * @brief Tokenizes the entire input.
   * @return A vector of all tokens.
   */
  std::vector<Token> tokenizeAll();

private:
  /** @brief Checks if the end of the file has been reached.
   * @return True if the end of the file is reached, false otherwise.
   */
  bool eof() const;

  /** @brief Returns the next character without advancing the position.
   * @return The next character.
   */
  char peek() const;

  /** @brief Returns the next character and advances the position.
   * @return The next character.
   */
  char get();

  /**
   * @brief Identifies if a character is a valid identifier start.
   * @param ch The character to check.
   * @return True if the character can start an identifier, false otherwise.
   */

  bool isIdentifier(char ch) const;
  /**
   * @brief Identifies if a character is a valid symbol.
   * @param ch The character to check.
   * @return True if the character is a symbol, false otherwise.
   */
  bool isSymbol(char ch) const;

  /**
   * @brief Reads an identifier token from the input.
   * @param start The starting line and column of the token.
   * @return The identifier token.
   */
  Token readIdentifier(std::pair<size_t, size_t> start);

  /**
   * @brief Reads a number token from the input.
   * @param start The starting line and column of the token.
   * @return The number token.
   */
  Token readNumber(std::pair<size_t, size_t> start);

  /**
   * @brief Reads a string token from the input.
   * @param start The starting line and column of the token.
   * @return The string token.
   */
  Token readString(std::pair<size_t, size_t> start);

  /**
   * @brief Reads a symbol token from the input.
   * @param start The starting line and column of the token.
   * @return The symbol token.
   */
  Token readSymbol(std::pair<size_t, size_t> start);

private:
  std::ifstream _file;
  std::string _buffer;
  size_t _position;
  size_t _line;
  size_t _column;
};

/**
 * @brief Outputs a token to the given output stream.
 * @param os The output stream.
 * @param token The token to output.
 * @return The output stream.
 */
std::ostream &operator<<(std::ostream &os, const Token &token);

} // namespace picceler