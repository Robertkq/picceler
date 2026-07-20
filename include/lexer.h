/**
 * @file lexer.h
 * @brief Lexer for the picceler programming language.
 *
 * The lexer tokenizes strings from a source file into tokens for parsing.
 *
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "error.h"

namespace picceler {

/**
 * @struct Token
 * @brief Represents a token produced by the lexer.
 */
struct Token {
  /** @brief The type of the token. */
  enum class Type : uint8_t {
    IDENTIFIER, // Represents identifiers (user defined names)
    NUMBER,     // Represents numeric literals (e.g., integers, floats)
    STRING,     // Represents string literals
    L_PAREN,    // Represents the left parenthesis '('
    R_PAREN,    // Represents the right parenthesis ')'
    L_BRACKET,  // Represents the left bracket '['
    R_BRACKET,  // Represents the right bracket ']'
    L_BRACE,    // Represents the left brace '{'
    R_BRACE,    // Represents the right brace '}'
    COMMA,      // Represents the comma ','
    COLON,      // Represents the colon ':'
    ARROW,      // Represents the arrow '->'
    ASSIGN,     // Represents the assignment operator '='
    TYPE,       // Represents type annotations (e.g., int, float, string)
    KW_DEF,     // Represents the keyword 'def'
    KW_RETURN,  // Represents the keyword 'return'
    EOF_TOKEN,  // Represents the end of file
    UNKNOWN     // Represents unknown tokens
  };

  Token() : _type(Type::UNKNOWN), _value(""), _location() {}

  Token(Type type, std::string value, Location location) : _type(type), _value(std::move(value)), _location(location) {}

  Token(const Token &other) : _type(other._type), _value(other._value), _location(other._location) {}
  Token(Token &&other) noexcept : _type(other._type), _value(std::move(other._value)), _location(other._location) {}
  ~Token() = default;
  Token &operator=(const Token &other) {
    if (this != &other) {
      _type = other._type;
      _value = other._value;
      _location = other._location;
    }
    return *this;
  }
  Token &operator=(Token &&other) noexcept {
    if (this != &other) {
      _type = other._type;
      _value = std::move(other._value);
      _location = other._location;
    }
    return *this;
  }

  /**
   * @brief Converts the token type to a string representation.
   * @return The string representation of the token type.
   */
  std::string typeToString() const;

  /**
   * @brief Converts the token to a string representation.
   * @return The string representation of the token.
   */
  std::string toString() const {
    return std::format("Token(type: {}, value: '{}', line: {}, column: {})", typeToString(), _value, _location.line(),
                       _location.column());
  }

  Type type() const { return _type; }
  const std::string &value() const { return _value; }
  size_t line() const { return _location.line(); }
  size_t column() const { return _location.column(); }
  Location location() const { return _location; }
  /*
   * @brief Compares a token to a token type for easy checking in parsing.
   * @param lhs The token to compare.
   * @param rhs The token type to compare against.
   * @return True if the token's type matches the token type, false otherwise.
   */
  friend constexpr bool operator==(const Token &lhs, const Token::Type &rhs) { return lhs._type == rhs; }

private:
  Type _type;
  std::string _value;
  Location _location;
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
  ////////
  void setSourceString(std::string_view source);
  ////////
  /**
   * @brief Sets the source file for the lexer.
   * @param source The source file to read from.
   */
  Result<void> setSource(const std::string &filepath);

  /**
   * @brief Returns the next token from the input.
   * @return The next token.
   */
  Result<Token> nextToken();

  /**
   * @brief Returns the next token without advancing the input.
   * @return The next token.
   */
  Result<Token> peekToken();

  /**
   * @brief Skips whitespace characters in the input.
   */
  void skipWhitespace();

  /**
   * @brief Tokenizes the entire input.
   * @return A vector of all tokens.
   */
  Result<std::vector<Token>> tokenizeAll();

private:
  /////////
  void resetState();
  /////////
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
   * @brief Identifies if a string is a keyword.
   * @param value The string to check.
   * @return The token type if the string is a keyword, std::unexpected otherwise.
   */
  Result<Token::Type> isKeyword(const std::string &value) const;

  /**
   * @brief Identifies if a string is a type.
   * @param value The string to check.
   * @return The token type if the string is a type, std::unexpected otherwise.
   */
  Result<Token::Type> isType(const std::string &value) const;

  /**
   * @brief Reads an identifier, keyword or type token from the input.
   * @param start The starting line and column of the token.
   * @return The identifier, keyword or type token.
   */
  Result<Token> readIdentifierOrKeywordOrType(std::pair<size_t, size_t> start);

  /**
   * @brief Reads a number token from the input.
   * @param start The starting line and column of the token.
   * @return The number token.
   */
  Result<Token> readNumber(std::pair<size_t, size_t> start);

  /**
   * @brief Reads a string token from the input.
   * @param start The starting line and column of the token.
   * @return The string token.
   */
  Result<Token> readString(std::pair<size_t, size_t> start);

  /**
   * @brief Reads a symbol token from the input.
   * @param start The starting line and column of the token.
   * @return The symbol token.
   */
  Result<Token> readSymbol(std::pair<size_t, size_t> start);

private:
  std::string unescapeString(std::string &&string) const;

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