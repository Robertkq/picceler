#pragma once

#include <expected>
#include <string>
#include <format>

namespace picceler {

/**
 * @brief Struct to hold compilation error information.
 */
struct CompileError {

  CompileError() : _message("Empty message! If this isn't a test, debug this."), _line(0), _column(0) {}
  explicit CompileError(std::string &&string, size_t line = 0, size_t column = 0)
      : _message(std::move(string)), _line(line), _column(column) {}
  explicit CompileError(const std::string &string, size_t line = 0, size_t column = 0)
      : _message(string), _line(line), _column(column) {}

  CompileError(const CompileError &) = default;
  CompileError(CompileError &&) = default;

  CompileError &operator=(const CompileError &) = default;
  CompileError &operator=(CompileError &&) = default;

  std::string _message;
  size_t _line;
  size_t _column;

  //   const std::string &message() const { return _message; }
  size_t line() const { return _line; }
  size_t column() const { return _column; }

  std::string message() const {
    if (_line == 0 && _column == 0) {
      return std::format("CompilerError: {}", _message);
    }

    return std::format("[line {}:{}] CompilerError: {}", _line, _column, _message);
  }
};

/**
 * @brief A type alias for the result of a compilation operation.
 * @tparam T The type of the successful result.
 */
template <typename T> using Result = std::expected<T, CompileError>;

} // namespace picceler
