#include "ast.h"

#include "spdlog/spdlog.h"

#include <format>
#include <algorithm>
namespace picceler {

std::string ModuleNode::toString() const { return std::format("Module: {} statements", statements().size()); }

void ModuleNode::normalizeTopLevelStatements() {
  spdlog::info("Printing all statements in the AST:\n");
  for (auto stmt : statements()) {
    spdlog::info("{}\n", stmt->toString());
  }
  auto modified = wrapTopLevelStatementsInMain();
  if (modified) {
    spdlog::info("AST normalization modified the graph. Final AST:\n");
    for (auto stmt : statements()) {
      spdlog::info("{}\n", stmt->toString());
    }
  }
}

bool ModuleNode::wrapTopLevelStatementsInMain() {
  bool hasMain = false;
  bool onlyFunctions = true;

  for (auto stmt : statements()) {
    if (auto funcNode = dynamic_cast<FunctionNode *>(stmt)) {
      if (funcNode->name() == "main") {
        hasMain = true;
      }
    } else {
      onlyFunctions = false;
    }
  }

  if (!hasMain) {
    if (!onlyFunctions) {
      spdlog::warn("No 'main' functions found, implicit 'main' will be generated to wrap the top-level statements");
      auto mainFunc = std::make_unique<FunctionNode>("main");
      for (auto &stmt : _statements) {
        auto stmtPtr = stmt.get();
        if (!dynamic_cast<FunctionNode *>(stmtPtr)) {
          mainFunc->addBodyStatement(std::move(stmt));
        }
      }
      _statements.erase(std::remove_if(_statements.begin(), _statements.end(),
                                       [&](const std::unique_ptr<ASTNode> &stmt) {
                                         return !dynamic_cast<FunctionNode *>(stmt.get());
                                       }),
                        _statements.end());
      _statements.push_back(std::move(mainFunc));
    } else {
      spdlog::error("No 'main' function found and no top-level statements to wrap.");
      return false;
    }
  } else {
    if (!onlyFunctions) {
      spdlog::warn("Found 'main' function, but also found top-level statements. Not allowed.");
      return false;
    }
  }

  return true;
}

std::string FunctionNode::toString() const {
  std::string params;
  for (const auto &param : parameters()) {
    if (!params.empty()) {
      params += ", ";
    }
    params += std::format("{}: {}", param.first, param.second);
  }
  std::string statements;
  if (!body().empty()) {
    for (const auto &statement : body()) {
      statements += "\t" + statement->toString() + "\n";
    }
  }
  return std::format("Function:[{}({})] {{\n{}}}", name(), params, statements);
}

std::string VariableNode::toString() const { return std::format("Variable:[{}]", name()); }

std::string StringNode::toString() const { return std::format("String:[{}]", value()); }

std::string NumberNode::toString() const { return std::format("Number:[{}]", value()); }

std::string AssignmentNode::toString() const {
  return std::format("Assignment:[{} = {}]", lhs()->toString(), rhs()->toString());
}

std::string CallNode::toString() const {
  std::string args;
  for (const auto &arg : arguments()) {
    if (!args.empty()) {
      args += ", ";
    }
    args += arg->toString();
  }
  return std::format("Call:[{}({})]", callee(), args);
}

std::string KernelNode::toString() const {
  std::string repr = std::format("Kernel({}x{}):[", rows().size(), rows().empty() ? 0 : rows()[0].size());
  for (const auto &row : rows()) {
    repr += " [";
    for (size_t i = 0; i < row.size(); ++i) {
      repr += std::to_string(row[i]);
      if (i < row.size() - 1) {
        repr += ", ";
      }
    }
    repr += "], ";
  }
  repr += "]";
  return repr;
}

} // namespace picceler