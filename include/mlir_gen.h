#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"

#include "dialect.h"
#include "types.h"
#include "parser.h"

namespace picceler {

/**
 * @brief MLIR code generator that converts AST nodes to MLIR constructs.
 * This offers the initial IR generation from the AST.
 */
class MLIRGen {

public:
  using GeneratorFunction = std::function<mlir::Value(mlir::Location, const std::vector<mlir::Value> &)>;

public:
  MLIRGen(mlir::MLIRContext *context);

  bool normalizeASTMain(mlir::ModuleOp module, ModuleNode *root);
  void declareUserFunctions(mlir::ModuleOp module, ModuleNode *root);
  void defineUserFunctions(mlir::ModuleOp module, ModuleNode *root);

  std::vector<mlir::Type> getFunctionArgTypes(FunctionNode *funcNode);

  /**
   * @brief Generates MLIR code from the given AST root node.
   * @param root The root node of the AST.
   * @return The generated MLIR module operation.
   */
  mlir::ModuleOp generate(ModuleNode *root);

private:
  /**
   * @brief Emits MLIR code for a given AST node.
   * @param node The AST node to emit from.
   */
  void emitStatement(ASTNode *node);

  /**
   * @brief Helper functions to emit specific AST node types.
   * \{
   */
  mlir::Value emitKernel(KernelNode *node);
  mlir::Value emitExpression(ASTNode *node);
  mlir::Value emitAssignment(AssignmentNode *node);
  mlir::Value emitCall(CallNode *node);
  mlir::Value emitVariable(VariableNode *node);
  mlir::Value emitString(StringNode *node);
  mlir::Value emitNumber(NumberNode *node);
  mlir::Value emitCallExpression(CallNode *node, const std::vector<mlir::Value> &args);

  /**
   * \}
   */

  void registerBuiltinFunctions();

private:
  mlir::MLIRContext *_context;
  mlir::OpBuilder _builder;
  std::unordered_map<std::string, mlir::Value> _variableTable;
  std::unordered_map<std::string, GeneratorFunction> _functionTable;
};

} // namespace picceler