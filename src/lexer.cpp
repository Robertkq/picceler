#include "lexer.h"

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
  case Type::L_PAREN:
    return "L_PAREN";
  case Type::R_PAREN:
    return "R_PAREN";
  case Type::L_BRACKET:
    return "L_BRACKET";
  case Type::R_BRACKET:
    return "R_BRACKET";
  case Type::L_BRACE:
    return "L_BRACE";
  case Type::R_BRACE:
    return "R_BRACE";
  case Type::COMMA:
    return "COMMA";
  case Type::COLON:
    return "COLON";
  case Type::ARROW:
    return "ARROW";
  case Type::ASSIGN:
    return "ASSIGN";
  case Type::TYPE:
    return "TYPE";
  case Type::KW_DEF:
    return "KW_DEF";
  case Type::KW_RETURN:
    return "KW_RETURN";
  case Type::EOF_TOKEN:
    return "EOF_TOKEN";
  default:
    return "UNKNOWN";
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
    return Token{Token::Type::EOF_TOKEN, "", Location{{_line, _column}}};
  }

  char ch;
  ch = peek();
  if (isIdentifier(ch)) {
    return readIdentifierOrKeywordOrType({_line, _column});
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
  return Token{Token::Type::UNKNOWN, "", Location{{_line, _column}}};
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
  } while (*token != Token::Type::EOF_TOKEN);
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
  static const std::string symbols = "=():,[]{}->=";
  return symbols.find(ch) != std::string::npos;
}

Result<Token::Type> Lexer::isKeyword(const std::string &value) const {
  static const std::unordered_map<std::string, Token::Type> keywords = {
      {"def", Token::Type::KW_DEF},
      {"return", Token::Type::KW_RETURN},
  };
  auto it = keywords.find(value);
  if (it != keywords.end()) {
    return it->second;
  }
  return std::unexpected(CompileError{std::format("Not a keyword: {}", value), Location{_line, _column}});
}

Result<Token::Type> Lexer::isType(const std::string &value) const {
  static const std::unordered_map<std::string, Token::Type> types = {
      {"int64", Token::Type::TYPE},
      {"f64", Token::Type::TYPE},
      {"string", Token::Type::TYPE},
      {"image", Token::Type::TYPE},
  };
  auto it = types.find(value);
  if (it != types.end()) {
    return it->second;
  }
  return std::unexpected(CompileError{std::format("Not a type: {}", value), Location{_line, _column}});
}

Result<Token> Lexer::readIdentifierOrKeywordOrType(std::pair<size_t, size_t> start) {
  std::string value;
  while (!eof() && (isalnum(peek()) || peek() == '_')) {
    value += get();
  }
  auto result = isKeyword(value);
  if (result) {
    Token::Type keywordType = *result;
    return Token{keywordType, value, Location{start}};
  }
  result = isType(value);
  if (result) {
    Token::Type typeType = *result;
    return Token{typeType, value, Location{start}};
  }
  return Token{Token::Type::IDENTIFIER, value, Location{start}};
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
        return std::unexpected(CompileError{"Invalid number format: multiple decimal points", Location{start}});
      }
      hasDot = true;
      value += get();
    } else {
      break;
    }
  }
  return Token{Token::Type::NUMBER, value, Location{start}};
}

Result<Token> Lexer::readString(std::pair<size_t, size_t> start) {
  get(); // consume the opening quote

  std::string value;
  while (!eof() && peek() != '"') {
    value += get();
  }
  value = unescapeString(std::move(value));
  if (!eof())
    get(); // consume the closing quote
  return Token{Token::Type::STRING, value, Location{start}};
}

Result<Token> Lexer::readSymbol(std::pair<size_t, size_t> start) {
  char ch = get();
  switch (ch) {
  case '(':
    return Token{Token::Type::L_PAREN, "(", Location{start}};
  case ')':
    return Token{Token::Type::R_PAREN, ")", Location{start}};
  case '[':
    return Token{Token::Type::L_BRACKET, "[", Location{start}};
  case ']':
    return Token{Token::Type::R_BRACKET, "]", Location{start}};
  case '{':
    return Token{Token::Type::L_BRACE, "{", Location{start}};
  case '}':
    return Token{Token::Type::R_BRACE, "}", Location{start}};
  case ',':
    return Token{Token::Type::COMMA, ",", Location{start}};
  case ':':
    return Token{Token::Type::COLON, ":", Location{start}};
  case '-':
    if (!eof() && peek() == '>') {
      get(); // consume '>'
      return Token{Token::Type::ARROW, "->", Location{start}};
    }
    return Token{Token::Type::UNKNOWN, std::string(1, ch), Location{start}};
  case '=':
    return Token{Token::Type::ASSIGN, "=", Location{start}};
  default:
    return Token{Token::Type::UNKNOWN, std::string(1, ch), Location{start}};
  }
}

std::string Lexer::unescapeString(std::string &&string) const {
  std::string unescaped;
  unescaped.reserve(string.size());

  for (size_t i = 0; i < string.size(); ++i) {
    if (string[i] == '\\' && i + 1 < string.size()) {
      switch (string[i + 1]) {
      case 'n':
        unescaped += '\n';
        break; // Newline (0x0A)
      case 't':
        unescaped += '\t';
        break; // Tab (0x09)
      case 'r':
        unescaped += '\r';
        break; // Carriage Return (0x0D)
      case '\\':
        unescaped += '\\';
        break; // Literal Backslash
      case '"':
        unescaped += '"';
        break; // Double Quote
      case '\'':
        unescaped += '\'';
        break; // Single Quote
      case '0':
        unescaped += '\0';
        break; // Null Byte
      default:
        // If it's an unrecognized escape (e.g. \a), keep the backslash and character
        unescaped += '\\';
        unescaped += string[i + 1];
        break;
      }
      ++i;
    } else {
      unescaped += string[i];
    }
  }

  return unescaped;
}

std::ostream &operator<<(std::ostream &os, const Token &token) {
  os << token.toString();
  return os;
}

} // namespace picceler