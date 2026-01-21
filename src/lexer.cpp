#include "lexer.h"

#include <iostream>
#include <format>

#include "spdlog/spdlog.h"

namespace picceler {

std::string Token::typeToString() const {
  switch (_type) {
  case Type::IDENTIFIER:
    return "IDENTIFIER";
  case Type::NUMBER:
    return "NUMBER";
  case Type::STRING:
    return "STRING";
  case Type::SYMBOL:
    return "SYMBOL";
  case Type::EOF_TOKEN:
    return "EOF";
  case Type::UNKNOWN:
    return "UNKNOWN";
  default:
    spdlog::error("Invalid token type");
    return "INVALID";
  }
}

Lexer::Lexer() : _file(), _buffer(), _position(0), _line(1), _column(1) {}

Result<void> Lexer::setSource(const std::string &source) {
  if (_file.is_open()) {
    _file.close();
  }
  _file.open(source);
  if (!_file.is_open()) {
    return std::unexpected(CompileError(std::format("Cannot open source file {}", source)));
  }
  _buffer.assign((std::istreambuf_iterator<char>(_file)), std::istreambuf_iterator<char>());
  _file.close();
  _position = 0;
  _line = 1;
  _column = 1;

  return {};
}

Result<Token> Lexer::nextToken() {
  skipWhitespace();

  if (eof()) {
    return Token{Token::Type::EOF_TOKEN, "", _line, _column};
  }

  char ch;
  ch = peek();
  if (isIdentifier(ch)) {
    return readIdentifier({_line, _column});
  }
  if (isdigit(ch) || ch == '-') {
    return readNumber({_line, _column});
  }
  if (ch == '"') {
    return readString({_line, _column});
  }
  if (isSymbol(ch)) {
    return readSymbol({_line, _column});
  }

  get(); // consume unknown character
  return Token{Token::Type::UNKNOWN, "", _line, _column};
}

Result<Token> Lexer::peekToken() {
  size_t oldPos = _position;
  size_t oldLine = _line;
  size_t oldCol = _column;
  auto token = nextToken();
  _position = oldPos;
  _line = oldLine;
  _column = oldCol;
  return token;
}

void Lexer::skipWhitespace() {
  while (!eof() && isspace(peek())) {
    get();
  }
}

Result<std::vector<Token>> Lexer::tokenizeAll() {
  std::vector<Token> tokens;
  Result<Token> token;
  do {
    token = nextToken();
    if (token) {
      tokens.push_back(*token);
    } else {
      spdlog::error(token.error().message());
      return std::unexpected(CompileError{"Failed to tokenize the entire source file"});
    }
  } while (token.value()._type != Token::Type::EOF_TOKEN);
  return tokens;
}

bool Lexer::eof() const { return _position >= _buffer.size(); }

char Lexer::peek() const { return eof() ? '\0' : _buffer[_position]; }

char Lexer::get() {
  if (eof()) {
    throw std::runtime_error("Attempt to read past end of buffer");
  }
  if (_buffer[_position] == '\n') {
    _line++;
    _column = 1;
  } else {
    _column++;
  }
  return _buffer[_position++];
}

bool Lexer::isIdentifier(char ch) const { return isalpha(ch) || ch == '_'; }

bool Lexer::isSymbol(char ch) const {
  static const std::string _symbols = "=():,[]";
  return _symbols.find(ch) != std::string::npos;
}

Result<Token> Lexer::readIdentifier(std::pair<size_t, size_t> start) {
  std::string value;
  while (!eof() && (isalnum(peek()) || peek() == '_')) {
    value += get();
  }
  return Token{Token::Type::IDENTIFIER, value, start.first, start.second};
}

Result<Token> Lexer::readNumber(std::pair<size_t, size_t> start) {
  std::string value;
  bool hasDot = false;
  if (peek() == '-') {
    value += get();
  }

  while (!eof()) {
    char ch = peek();
    if (isdigit(ch)) {
      value += get();
    } else if (ch == '.') {
      if (hasDot) {
        return std::unexpected(
            CompileError{"Invalid number format: multiple decimal points", start.first, start.second});
      }
      hasDot = true;
      value += get();
    } else {
      break;
    }
  }
  return Token{Token::Type::NUMBER, value, start.first, start.second};
}

Result<Token> Lexer::readString(std::pair<size_t, size_t> start) {
  get(); // consume the opening quote

  std::string value;
  while (!eof() && peek() != '"') {
    value += get();
  }
  if (!eof())
    get(); // consume the closing quote
  return Token{Token::Type::STRING, value, start.first, start.second};
}

Result<Token> Lexer::readSymbol(std::pair<size_t, size_t> start) {
  char ch = get();
  return Token{Token::Type::SYMBOL, std::string(1, ch), start.first, start.second};
}

std::ostream &operator<<(std::ostream &os, const Token &token) {
  os << token.toString();
  return os;
}

} // namespace picceler