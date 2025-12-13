#include <format>

#include "ast.h"

namespace picceler {

std::string ModuleNode::toString() const {
  return std::format("Module: {} statements", statements.size());
}

std::string VariableNode::toString() const {
  return std::format("Variable:[{}]", name);
}

std::string StringNode::toString() const {
  return std::format("String:[{}]", value);
}

std::string NumberNode::toString() const {
  return std::format("Number:[{}]", value);
}

std::string AssignmentNode::toString() const {
  return std::format("Assignment:[{} = {}]", lhs->toString(), rhs->toString());
}

std::string CallNode::toString() const {
  std::string args;
  for (const auto &arg : arguments) {
    if (!args.empty()) {
      args += ", ";
    }
    args += arg->toString();
  }
  return std::format("Call:[{}({})]", callee, args);
}

} // namespace picceler