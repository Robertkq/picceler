#pragma once

#include <expected>
#include <string>
#include <format>

namespace picceler {

/** @brief Struct to hold location information. */
class Location {
public:
  Location() : _line(0), _column(0) {}
  explicit Location(std::pair<size_t, size_t> line_column) : _line(line_column.first), _column(line_column.second) {}
  explicit Location(size_t line, size_t column) : _line(line), _column(column) {}

  size_t line() const { return _line; }
  size_t column() const { return _column; }

private:
  size_t _line = 0;
  size_t _column = 0;
};

/**
 * @brief Struct to hold compilation error information.
 */
struct CompileError {

  CompileError() : _message("Empty message! If this isn't a test, debug this."), _location() {}
  explicit CompileError(std::string string, Location location = {})
      : _message(std::move(string)), _location(location) {}

  CompileError(const CompileError &) = default;
  CompileError(CompileError &&) = default;

  CompileError &operator=(const CompileError &) = default;
  CompileError &operator=(CompileError &&) = default;

  std::string _message;
  Location _location;

  //   const std::string &message() const { return _message; }
  size_t line() const { return _location.line(); }
  size_t column() const { return _location.column(); }

  std::string message() const {
    if (_location.line() == 0 && _location.column() == 0) {
      return std::format("CompilerError: {}", _message);
    }

    return std::format("[line {}:{}] CompilerError: {}", _location.line(), _location.column(), _message);
  }
};

/**
 * @brief A type alias for the result of a compilation operation.
 * @tparam T The type of the successful result.
 */
template <typename T> using Result = std::expected<T, CompileError>;

} // namespace picceler
