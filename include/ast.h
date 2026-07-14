#pragma once

#include <vector>
#include <string>
#include <optional>
#include <memory>

#include "lexer.h"

namespace picceler {

template <typename T> std::vector<T *> getRawPointers(const std::vector<std::unique_ptr<T>> &vec) {
  std::vector<T *> rawPointers;
  rawPointers.reserve(vec.size());
  for (const auto &item : vec) {
    rawPointers.push_back(item.get());
  }
  return rawPointers;
}

/**
 * @brief Abstract Syntax Tree (AST) node base class.
 */
class ASTNode {
public:
  ASTNode() = default;
  virtual ~ASTNode() = default;

  ASTNode(const ASTNode &) = delete;
  ASTNode &operator=(const ASTNode &) = delete;
  ASTNode(ASTNode &&) = default;
  ASTNode &operator=(ASTNode &&) = default;

  virtual std::string toString() const = 0;
};

/**
 * @brief AST node for the entire module.
 */

class ModuleNode : public ASTNode {
public:
  std::vector<ASTNode *> statements() const { return getRawPointers(_statements); }
  std::vector<std::unique_ptr<ASTNode>> &mutableStatements() { return _statements; }

  void addStatement(std::unique_ptr<ASTNode> statement) { _statements.push_back(std::move(statement)); }

  std::string toString() const override;

private:
  std::vector<std::unique_ptr<ASTNode>> _statements;
};

/**
 * @brief AST node for function definitions.
 */
class FunctionNode : public ASTNode {
public:
  FunctionNode(std::string name) : _name(std::move(name)) {}

  const std::string &name() const { return _name; }
  const auto &parameters() const { return _parameters; }
  std::vector<ASTNode *> body() const { return getRawPointers(_body); }

  void addParameter(const std::string &paramName, const std::string &paramType) {
    _parameters.emplace_back(paramName, paramType);
  }

  void addBodyStatement(std::unique_ptr<ASTNode> statement) { _body.push_back(std::move(statement)); }

  std::string toString() const override;

private:
  std::string _name;
  std::vector<std::pair<std::string, std::string>> _parameters;
  std::vector<std::unique_ptr<ASTNode>> _body;
};

/**
 * @brief AST node for variable references.
 */
class VariableNode : public ASTNode {
public:
  VariableNode(std::string name, std::optional<std::string> type = std::nullopt)
      : _name(std::move(name)), _type(std::move(type)) {}

  const std::string &name() const { return _name; }
  const std::optional<std::string> &type() const { return _type; }

  std::string toString() const override;

private:
  std::string _name;
  std::optional<std::string> _type; // optional type annotation
};

/**
 * @brief AST node for string literals.
 */
class StringNode : public ASTNode {
public:
  StringNode(std::string value) : _value(std::move(value)) {}

  const std::string &value() const { return _value; }

  std::string toString() const override;

private:
  std::string _value;
};

/**
 * @brief AST node for numeric literals.
 */
class NumberNode : public ASTNode {
public:
  NumberNode(double value) : _value(value) {}

  double value() const { return _value; }

  std::string toString() const override;

private:
  double _value;
};

/**
 * @brief AST node for assignment statements.
 */
class AssignmentNode : public ASTNode {
public:
  AssignmentNode(std::unique_ptr<VariableNode> lhs, std::unique_ptr<ASTNode> rhs)
      : _lhs(std::move(lhs)), _rhs(std::move(rhs)) {}

  VariableNode *lhs() const { return _lhs.get(); }
  ASTNode *rhs() const { return _rhs.get(); }

  std::string toString() const override;

private:
  std::unique_ptr<VariableNode> _lhs;
  std::unique_ptr<ASTNode> _rhs;
};

/**
 * @brief AST node for function calls.
 */
class CallNode : public ASTNode {
public:
  CallNode(std::string callee) : _callee(std::move(callee)), _arguments() {}

  const std::string &callee() const { return _callee; }
  std::vector<ASTNode *> arguments() const { return getRawPointers(_arguments); }

  void addArgument(std::unique_ptr<ASTNode> arg) { _arguments.push_back(std::move(arg)); }

  std::string toString() const override;

private:
  std::string _callee;
  std::vector<std::unique_ptr<ASTNode>> _arguments;
};

/**
 * @brief AST node for kernel definitions.
 */

class KernelNode : public ASTNode {
public:
  const std::vector<std::vector<double>> &rows() const { return _rows; }

  void addRow(const std::vector<double> &row) { _rows.push_back(row); }

  std::string toString() const override;

private:
  std::vector<std::vector<double>> _rows;
};

} // namespace picceler
