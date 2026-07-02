#pragma once

#include <mlir/IR/MLIRContext.h>
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
  using VariableTable = std::unordered_map<std::string, mlir::Value>;
  using NamedVariableTable = std::pair<std::string, VariableTable>;

public:
  MLIRGen() = delete;
  MLIRGen(mlir::MLIRContext *context);
  MLIRGen(const MLIRGen &) = default;
  MLIRGen &operator=(const MLIRGen &) = default;
  MLIRGen(MLIRGen &&) = default;
  MLIRGen &operator=(MLIRGen &&) = default;
  ~MLIRGen();

  /**
   * @brief Generates MLIR code from the given AST root node.
   * @param root The root node of the AST.
   * @return The generated MLIR module operation.
   */
  mlir::ModuleOp generate(ModuleNode *root);

private:
  /**
   * @brief Normalizes the AST by processing all top-level statements in the module and assign to main if not found.
   * @param module The MLIR module operation.
   * @param root The root node of the AST.
   * @return True if normalization is successful, false otherwise.
   */
  bool normalizeASTMain(mlir::ModuleOp module, ModuleNode *root);

  /**
   * @brief Declares user-defined functions in the MLIR module.
   * @param module The MLIR module operation.
   * @param root The root node of the AST.
   */
  void declareUserFunctions(mlir::ModuleOp module, ModuleNode *root);

  /**
   * @brief Defines user-defined functions in the MLIR module.
   * @param module The MLIR module operation.
   * @param root The root node of the AST.
   */
  void defineUserFunctions(mlir::ModuleOp module, ModuleNode *root);

  /**
   *    @brief Retrieves the argument types for a given function node.
   *    @param funcNode The function node to retrieve argument types from.
   *    @return A vector of MLIR types representing the argument types of the function.
   */
  std::vector<mlir::Type> getFunctionArgTypes(FunctionNode *funcNode);

  /**
   * @brief Looks up a variable in the known scopes and returns its MLIR value if found.
   * @param name The name of the variable to look up.
   * @return The MLIR value of the variable if found, or an error if not found.
   */
  Result<mlir::Value> lookupVariable(const std::string &name) const;

  /**
   * @brief Declares a variable in the current scope with the given name and MLIR value.
   * @param name The name of the variable to declare.
   * @param value The MLIR value of the variable.
   */
  void declareVariable(const std::string &name, mlir::Value value);

  /**
   * @brief Enters a new scope with the given name.
   * @param name The name of the scope to enter.
   */
  void enterScope(const std::string &name);

  /**
   * @brief Exits the current scope.
   */
  void exitScope();

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

  /**
   * @brief Registers all builtin functions with the generator.
   */
  void registerBuiltinFunctions();

private:
  mlir::MLIRContext *_context;
  mlir::OpBuilder _builder;
  std::vector<NamedVariableTable> _scopedVariableTable;
  std::unordered_map<std::string, GeneratorFunction> _functionTable;
};

} // namespace picceler