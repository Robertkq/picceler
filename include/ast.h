#pragma once

#include "lexer.h"
#include <format>

namespace picceler {

/**
 * @brief Abstract Syntax Tree (AST) node base class.
 */
struct ASTNode {
  virtual ~ASTNode() = default;
  virtual std::string toString() const = 0;
};

/**
 * @brief AST node for the entire module.
 */

struct ModuleNode : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> statements;
  std::string toString() const override;
};

/**
 * @brief AST node for variable references.
 */
struct VariableNode : public ASTNode {
  std::string name;
  std::string toString() const override;
};

/**
 * @brief AST node for string literals.
 */
struct StringNode : public ASTNode {
  std::string value;
  std::string toString() const override;
};

/**
 * @brief AST node for numeric literals.
 */
struct NumberNode : public ASTNode {
  size_t value;
  std::string toString() const override;
};

/**
 * @brief AST node for assignment statements.
 */
struct AssignmentNode : public ASTNode {
  std::unique_ptr<VariableNode> lhs;
  std::unique_ptr<ASTNode> rhs;
  std::string toString() const override;
};

/**
 * @brief AST node for function calls.
 */
struct CallNode : public ASTNode {
  std::string callee;
  std::vector<std::unique_ptr<ASTNode>> arguments;
  std::string toString() const override;
};

} // namespace picceler
