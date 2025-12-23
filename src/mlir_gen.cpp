#include "mlir_gen.h"

#include <unordered_set>

#include "spdlog/spdlog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "dialect.h"
#include "ops.h"
#include "types.h"

namespace picceler {
MLIRGen::MLIRGen(mlir::MLIRContext *context)
    : _context(context), _builder(_context), _variableTable(),
      _functionTable() {}

mlir::ModuleOp MLIRGen::generate(ModuleNode *root) {
  auto module = mlir::ModuleOp::create(_builder.getUnknownLoc());
  // Create a top-level main function to own all generated ops.
  auto funcType = _builder.getFunctionType({}, {});
  auto mainFunc = _builder.create<mlir::func::FuncOp>(_builder.getUnknownLoc(),
                                                      "main", funcType);
  auto *entryBlock = mainFunc.addEntryBlock();
  _builder.setInsertionPointToStart(entryBlock);
  _variableTable = builtinVariables();
  _functionTable = builtinFunctions();

  for (auto &stmt : root->statements) {
    emitStatement(stmt.get());
  }

  // Terminate the function.
  _builder.create<mlir::func::ReturnOp>(_builder.getUnknownLoc());

  module.push_back(mainFunc);

  return module;
}

void MLIRGen::emitStatement(ASTNode *node) {
  if (auto assignment = dynamic_cast<AssignmentNode *>(node)) {
    emitAssignment(assignment);
  } else if (auto call = dynamic_cast<CallNode *>(node)) {
    emitCall(call);
  } else {
    throw std::runtime_error("Unsupported statement type");
  }
}

mlir::Value MLIRGen::emitExpression(ASTNode *node) {
  if (auto call = dynamic_cast<CallNode *>(node)) {
    return emitCall(call);
  } else if (auto var = dynamic_cast<VariableNode *>(node)) {
    return emitVariable(var);
  } else if (auto str = dynamic_cast<StringNode *>(node)) {
    return emitString(str);
  } else if (auto num = dynamic_cast<NumberNode *>(node)) {
    return emitNumber(num);
  } else {
    throw std::runtime_error("Unsupported expression type");
  }
}

/**
 * lhs - variable
 * rhs - expression
 */
mlir::Value MLIRGen::emitAssignment(AssignmentNode *node) {
  spdlog::debug("Emitting MLIR for assignment: {}", node->toString());
  auto rhs = emitExpression(node->rhs.get());
  _variableTable[node->lhs->name] = rhs; // Bind 'img' to %0
  return rhs;
}
mlir::Value MLIRGen::emitCall(CallNode *node) {
  spdlog::debug("Emitting MLIR for call: {}", node->toString());
  std::vector<mlir::Value> args;
  for (auto &arg : node->arguments) {
    args.push_back(emitExpression(arg.get()));
  }
  auto op = emitBuiltinCall(node, args);
  return op;
}

mlir::Value MLIRGen::emitVariable(VariableNode *node) {
  spdlog::debug("Emitting MLIR for variable: {}", node->toString());
  auto it = _variableTable.find(node->name);
  if (it != _variableTable.end()) {
    return it->second;
  }
  throw std::runtime_error("Undefined variable: " + node->name);
}

mlir::Value MLIRGen::emitString(StringNode *node) {
  spdlog::debug("Emitting MLIR for string: {}", node->toString());
  auto stringType = _builder.getType<StringType>();
  auto valueAttr = mlir::StringAttr::get(_context, node->value);
  return _builder.create<StringConstOp>(_builder.getUnknownLoc(), stringType,
                                        valueAttr);
}

mlir::Value MLIRGen::emitNumber(NumberNode *node) {
  spdlog::debug("Emitting MLIR for number: {}", node->toString());
  return _builder.create<mlir::arith::ConstantOp>(
      _builder.getUnknownLoc(),
      mlir::IntegerAttr::get(mlir::IntegerType::get(_context, 64),
                             node->value));
}

mlir::Value MLIRGen::emitBuiltinCall(CallNode *node,
                                     const std::vector<mlir::Value> &args) {
  const auto &name = node->callee;
  if (name == "load_image") {
    auto imageType = _builder.getType<ImageType>();
    auto filename = args[0];
    auto callOp = _builder.create<LoadImageOp>(_builder.getUnknownLoc(),
                                               imageType, filename);
    return callOp.getResult();
  } else if (name == "blur") {
    auto imageType = _builder.getType<ImageType>();
    auto &inputImage = args[0];
    auto &blurAmount = args[1];
    auto callOp = _builder.create<BlurOp>(_builder.getUnknownLoc(), imageType,
                                          inputImage, blurAmount);
    return callOp.getResult();
  } else if (name == "save_image") {
    auto &inputImage = args[0];
    auto &filename = args[1];
    _builder.create<SaveImageOp>(_builder.getUnknownLoc(), inputImage,
                                 filename);
    return {};
  } else if (name == "show_image") {
    auto &inputImage = args[0];
    _builder.create<ShowImageOp>(_builder.getUnknownLoc(), inputImage);
    return {};

  } else if (name == "brightness") {
    auto &inputImage = args[0];
    auto &brightnessValue = args[1];
    auto callOp = _builder.create<BrightnessOp>(_builder.getUnknownLoc(),
                                                inputImage.getType(),
                                                inputImage, brightnessValue);
    return callOp.getResult();
  } else if (name == "invert") {
    auto &inputImage = args[0];
    auto callOp = _builder.create<InvertOp>(_builder.getUnknownLoc(),
                                            inputImage.getType(), inputImage);
    return callOp.getResult();
  } else {
    throw std::runtime_error("Unsupported builtin function: " + name);
  }
}

std::unordered_map<std::string, mlir::Value> MLIRGen::builtinVariables() {
  auto stringType = StringType::get(_context);
  return {{"gaussian", _builder.create<StringConstOp>(
                           _builder.getUnknownLoc(), stringType,
                           mlir::StringAttr::get(_context, "gaussian"))}};
}

std::unordered_map<std::string, mlir::func::FuncOp>
MLIRGen::builtinFunctions() {
  return {};
}

} // namespace picceler