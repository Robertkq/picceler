#include "mlir_gen.h"

#include <cmath>
#include <limits>

#include "llvm/ADT/APFloat.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "ops.h"
#include "types.h"

namespace picceler {

/**
 * @brief Coerces a given MLIR value to a 64-bit integer if possible, with special handling for constants.
 * @param builder The MLIR OpBuilder to use for creating new operations if coercion is needed.
 * @param loc The location to use for any new operations created during coercion.
 * @param value The MLIR value to coerce.
 * @param opName The name of the operation for error reporting purposes.
 * @param argName The name of the argument for error reporting purposes.
 * @return The coerced MLIR value as a 64-bit integer.
 * @throws std::runtime_error if the value cannot be coerced to a 64-bit integer, or if a floating-point constant is not
 * a whole number or is out of range.
 */

mlir::Value coerceValueToInt64(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value, llvm::StringRef opName,
                               llvm::StringRef argName) {
  if (value.getType().isSignlessInteger(64)) {
    return value;
  }

  // Accept a non-I64 integer constant directly.
  if (auto constInt = value.getDefiningOp<mlir::arith::ConstantIntOp>()) {
    return builder.create<mlir::arith::ConstantIntOp>(loc, constInt.value(), 64);
  }

  // Accept a floating-point constant only if it is a whole number.
  if (auto constFloat = value.getDefiningOp<mlir::arith::ConstantFloatOp>()) {
    double numericValue = constFloat.value().convertToDouble();

    if (!std::isfinite(numericValue)) {
      throw std::runtime_error(opName.str() + " requires " + argName.str() + " to be a finite integer literal");
    }

    if (std::trunc(numericValue) != numericValue) {
      throw std::runtime_error(opName.str() + " requires " + argName.str() + " to be an integer literal");
    }

    if (numericValue < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        numericValue > static_cast<double>(std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error(opName.str() + " integer literal is out of range");
    }

    return builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(numericValue), 64);
  }

  // Anything else is not a literal, so we do not try to guess.
  throw std::runtime_error(opName.str() + " requires " + argName.str() + " to be a literal integer value");
}

MLIRGen::MLIRGen(mlir::MLIRContext *context)
    : _context(context), _builder(_context), _variableTable(), _functionTable() {}

/**
 * @brief Generates MLIR code from the given AST root node.
 * @param root The root node of the AST.
 * @return The generated MLIR module operation.
 */
mlir::ModuleOp MLIRGen::generate(ModuleNode *root) {
  auto module = mlir::ModuleOp::create(_builder.getUnknownLoc());
  // Create a top-level main function to own all generated ops.
  auto funcType = _builder.getFunctionType({}, {});
  auto mainFunc = _builder.create<mlir::func::FuncOp>(_builder.getUnknownLoc(), "main", funcType);
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

mlir::Value MLIRGen::emitKernel(KernelNode *node) {
  spdlog::debug("Emitting MLIR for kernel: {}", node->toString());

  int rows = node->rows.size();
  int cols = rows > 0 ? node->rows[0].size() : 0;

  if (rows == 0 || cols == 0) {
    throw std::runtime_error("Empty kernel is not allowed");
  }

  std::vector<double> flatValues;
  flatValues.reserve(rows * cols);
  for (const auto &row : node->rows) {
    flatValues.insert(flatValues.end(), row.begin(), row.end());
  }

  auto tensorType =
      mlir::RankedTensorType::get({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, _builder.getF64Type());

  auto valuesAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<double>(flatValues));

  auto resultType = KernelType::get(_context, rows, cols);

  auto op = _builder.create<KernelConstOp>(_builder.getUnknownLoc(), resultType, valuesAttr);

  return op.getResult();
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
  } else if (auto kernel = dynamic_cast<KernelNode *>(node)) {
    return emitKernel(kernel);
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
  return _builder.create<StringConstOp>(_builder.getUnknownLoc(), stringType, valueAttr);
}

mlir::Value MLIRGen::emitNumber(NumberNode *node) {
  spdlog::debug("Emitting MLIR for number: {}", node->toString());
  return _builder.create<mlir::arith::ConstantFloatOp>(_builder.getUnknownLoc(), _builder.getF64Type(),
                                                       llvm::APFloat(node->value));
}

mlir::Value MLIRGen::emitBuiltinCall(CallNode *node, const std::vector<mlir::Value> &args) {
  const auto &name = node->callee;
  if (name == "load_image") {
    auto imageType = _builder.getType<ImageType>();
    auto filename = args[0];
    auto callOp = _builder.create<LoadImageOp>(_builder.getUnknownLoc(), imageType, filename);
    return callOp.getResult();
  } else if (name == "save_image") {
    auto &inputImage = args[0];
    auto &filename = args[1];
    _builder.create<SaveImageOp>(_builder.getUnknownLoc(), inputImage, filename);
    return {};
  } else if (name == "show_image") {
    auto &inputImage = args[0];
    _builder.create<ShowImageOp>(_builder.getUnknownLoc(), inputImage);
    return {};
  } else if (name == "brightness") {
    auto &inputImage = args[0];
    auto brightnessValue = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "brightness", "value");
    auto callOp =
        _builder.create<BrightnessOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, brightnessValue);
    return callOp.getResult();
  } else if (name == "invert") {
    auto &inputImage = args[0];
    auto callOp = _builder.create<InvertOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage);
    return callOp.getResult();
  } else if (name == "convolution") {
    auto &inputImage = args[0];
    auto &kernel = args[1];
    auto callOp = _builder.create<ConvolutionOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, kernel);
    return callOp.getResult();
  } else if (name == "sharpen") {
    auto &inputImage = args[0];
    auto strengthValue = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "sharpen", "strength");
    auto callOp = _builder.create<SharpenOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, strengthValue);
    return callOp.getResult();
  } else if (name == "box_blur") {
    auto &inputImage = args[0];
    auto radiusValue = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "box_blur", "radius");
    auto callOp = _builder.create<BoxBlurOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, radiusValue);
    return callOp.getResult();
  } else if (name == "gaussian_blur") {
    auto &inputImage = args[0];
    auto radiusValue = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "gaussian_blur", "radius");
    auto callOp =
        _builder.create<GaussianBlurOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, radiusValue);
    return callOp.getResult();
  } else if (name == "edge_detect") {
    auto &inputImage = args[0];
    auto callOp = _builder.create<EdgeDetectOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage);
    return callOp.getResult();
  } else if (name == "emboss") {
    auto &inputImage = args[0];
    auto callOp = _builder.create<EmbossOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage);
    return callOp.getResult();
  } else if (name == "rotate") {
    auto &inputImage = args[0];
    auto angle = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "rotate", "angle");
    auto callOp = _builder.create<RotateOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, angle);
    return callOp.getResult();
  } else if (name == "crop") {
    auto &inputImage = args[0];
    auto x = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "crop", "x");
    auto y = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[2], "crop", "y");
    auto width = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[3], "crop", "width");
    auto height = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[4], "crop", "height");
    auto callOp =
        _builder.create<CropOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, x, y, width, height);
    return callOp.getResult();
  } else if (name == "diff") {
    auto &inputImage1 = args[0];
    auto &inputImage2 = args[1];
    auto callOp = _builder.create<DiffOp>(_builder.getUnknownLoc(), inputImage1.getType(), inputImage1, inputImage2);
    return callOp.getResult();
  } else if (name == "dilate") {
    auto &inputImage = args[0];
    auto radius = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "dilate", "radius");
    auto callOp = _builder.create<DilateOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, radius);
    return callOp.getResult();
  } else if (name == "erode") {
    auto &inputImage = args[0];
    auto radius = coerceValueToInt64(_builder, _builder.getUnknownLoc(), args[1], "erode", "radius");
    auto callOp = _builder.create<ErodeOp>(_builder.getUnknownLoc(), inputImage.getType(), inputImage, radius);
    return callOp.getResult();
  } else if (name == "blend") {
    auto &inputImage1 = args[0];
    auto &inputImage2 = args[1];
    auto &weight = args[2];
    auto callOp =
        _builder.create<BlendOp>(_builder.getUnknownLoc(), inputImage1.getType(), inputImage1, inputImage2, weight);
    return callOp.getResult();
  } else {
    throw std::runtime_error("Unsupported builtin function: " + name);
  }
}

std::unordered_map<std::string, mlir::Value> MLIRGen::builtinVariables() {
  auto variables = std::unordered_map<std::string, mlir::Value>{};
  auto stringType = StringType::get(_context);

  // variables["gaussian"] =
  //     _builder.create<StringConstOp>(_builder.getUnknownLoc(), stringType, mlir::StringAttr::get(_context,
  //     "gaussian"));

  return variables;
};

std::unordered_map<std::string, mlir::func::FuncOp> MLIRGen::builtinFunctions() {
  auto functions = std::unordered_map<std::string, mlir::func::FuncOp>{};
  return functions;
}

} // namespace picceler
