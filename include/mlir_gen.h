#pragma once

#include "dialect.h"
#include "parser.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/Pass.h>

namespace picceler {

class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext *context);

  mlir::ModuleOp generate(ModuleNode *root);

private:
  void emitStatement(ASTNode *node);
  mlir::Value emitExpression(ASTNode *node);
  mlir::Value emitAssignment(AssignmentNode *node);
  mlir::Value emitCall(CallNode *node);
  mlir::Value emitVariable(VariableNode *node);
  mlir::Value emitString(StringNode *node);
  mlir::Value emitNumber(NumberNode *node);

  bool isBuiltinFunction(CallNode *node);

  mlir::Value emitBuiltinCall(CallNode *node,
                              const std::vector<mlir::Value> &args);

  std::unordered_map<std::string, mlir::Value> builtinVariables();
  std::unordered_map<std::string, mlir::func::FuncOp> builtinFunctions();

private:
  mlir::MLIRContext *_context;
  mlir::OpBuilder _builder;
  std::unordered_map<std::string, mlir::Value> _variableTable;
  std::unordered_map<std::string, mlir::func::FuncOp> _functionTable;

  const std::string _initialOutputFile = "output.mlir";
};

} // namespace picceler