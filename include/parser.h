#pragma once

#include "lexer.h"
#include <format>

namespace picceler {

struct ASTNode {
  virtual ~ASTNode() = default;
  virtual std::string toString() const = 0;
};

struct ModuleNode : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> statements;
  std::string toString() const override {
    return std::format("Module: {} statements", statements.size());
  }
};

struct VariableNode : public ASTNode {
  std::string name;
  std::string toString() const override {
    return std::format("Variable:[{}]", name);
  }
};

struct StringNode : public ASTNode {
  std::string value;
  std::string toString() const override {
    return std::format("String:[{}]", value);
  }
};

struct NumberNode : public ASTNode {
  size_t value;
  std::string toString() const override {
    return std::format("Number:[{}]", value);
  }
};

struct AssignmentNode : public ASTNode {
  std::unique_ptr<VariableNode> lhs;
  std::unique_ptr<ASTNode> rhs;
  std::string toString() const override {
    return std::format("Assignment:[{} = {}]", lhs->toString(),
                       rhs->toString());
  }
};

struct CallNode : public ASTNode {
  std::string callee;
  std::vector<std::unique_ptr<ASTNode>> arguments;
  std::string toString() const override {
    std::string args;
    for (const auto &arg : arguments) {
      if (!args.empty()) {
        args += ", ";
      }
      args += arg->toString();
    }
    return std::format("Call:[{}({})]", callee, args);
  }
};

class Parser {
public:
  Parser();

  void setSource(const std::string &source);
  std::vector<Token> getTokens();

  std::unique_ptr<ModuleNode> parse();
  void printAST(const std::unique_ptr<ModuleNode> &node, int indent = 0);

private:
  std::unique_ptr<ASTNode> parseStatement();
  std::unique_ptr<ASTNode> parseExpression();
  std::unique_ptr<ASTNode> parseAssignment(Token identifier);
  std::unique_ptr<ASTNode> parseCall(Token identifier);
  std::unique_ptr<ASTNode> parseVariable(Token identifier = Token{
                                             Token::Type::UNKNOWN, "", 0, 0});
  std::unique_ptr<ASTNode> parseString();
  std::unique_ptr<ASTNode> parseNumber();

private:
  Lexer _lexer;
};

} // namespace picceler