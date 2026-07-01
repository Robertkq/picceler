#include "mlir_gen.h"

#include <cmath>
#include <limits>

#include "llvm/ADT/APFloat.h"

#include "spdlog/spdlog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"

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
  auto intBitWidth = 64;
  if (value.getType().isSignlessInteger(intBitWidth)) {
    return value;
  }

  // Accept a non-I64 integer constant directly.
  if (auto constInt = value.getDefiningOp<mlir::arith::ConstantIntOp>()) {
    return builder.create<mlir::arith::ConstantIntOp>(loc, constInt.value(), intBitWidth);
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

    return builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(numericValue), intBitWidth);
  }

  if (auto isRuntimeFloat = mlir::isa<mlir::FloatType>(value.getType())) {
    return builder.create<mlir::arith::FPToSIOp>(loc, builder.getI64Type(), value);
  }

  throw std::runtime_error(opName.str() + " requires " + argName.str() + ", it is an unsupported use case");
}

MLIRGen::MLIRGen(mlir::MLIRContext *context)
    : _context(context), _builder(_context), _scopedVariableTable(), _functionTable() {
  enterScope("Global"); // Enter the global scope
}

MLIRGen::~MLIRGen() {
  exitScope(); // Exit the global scope
}

bool MLIRGen::normalizeASTMain(mlir::ModuleOp module, ModuleNode *root) {
  bool hasMain = false;
  bool onlyFunctions = true;
  spdlog::info("Printing all statements in the AST:\n");
  for (const auto &stmt : root->statements) {
    spdlog::info("{}\n", stmt->toString());
  }

  for (const auto &stmt : root->statements) {
    if (auto funcNode = dynamic_cast<FunctionNode *>(stmt.get())) {
      if (funcNode->name == "main") {
        hasMain = true;
      }
    } else {
      onlyFunctions = false;
    }
  }

  if (!hasMain) {
    if (!onlyFunctions) {
      spdlog::warn("No 'main' functions found, implicit 'main' will be generated to wrap the top-level statements");
      auto mainFunc = std::make_unique<FunctionNode>();
      mainFunc->name = "main";
      for (auto &stmt : root->statements) {
        if (!dynamic_cast<FunctionNode *>(stmt.get())) {
          mainFunc->body.push_back(std::move(stmt));
        }
      }
      root->statements.erase(std::remove_if(root->statements.begin(), root->statements.end(),
                                            [&](const std::unique_ptr<ASTNode> &stmt) {
                                              return !dynamic_cast<FunctionNode *>(stmt.get());
                                            }),
                             root->statements.end());
      root->statements.push_back(std::move(mainFunc));
    } else {
      spdlog::error("No 'main' function found and no top-level statements to wrap.");
    }
  } else {
    if (!onlyFunctions) {
      spdlog::warn("Found 'main' function, but also found top-level statements. Not allowed.");
      return false;
    }
  }

  spdlog::info("AST normalization complete. Final AST:\n");
  for (const auto &stmt : root->statements) {
    spdlog::info("{}\n", stmt->toString());
  }
  return true;
}

void MLIRGen::declareUserFunctions(mlir::ModuleOp module, ModuleNode *root) {
  spdlog::debug("Declaring user-defined functions in the MLIR module");
  _builder.setInsertionPointToStart(module.getBody());
  for (const auto &stmt : root->statements) {
    if (auto funcNode = dynamic_cast<FunctionNode *>(stmt.get())) {
      spdlog::debug("Declaring function: {}", funcNode->name);
      std::vector<mlir::Type> funcArgTypes = getFunctionArgTypes(funcNode);
      auto funcType = _builder.getFunctionType(funcArgTypes, {});
      auto funcOp = _builder.create<mlir::func::FuncOp>(_builder.getUnknownLoc(), funcNode->name, funcType);
      _functionTable[funcNode->name] = [&, funcOp](mlir::Location loc, const std::vector<mlir::Value> &args) {
        auto callOp = _builder.create<mlir::func::CallOp>(loc, funcOp, args);
        return callOp.getNumResults() > 0 ? callOp.getResult(0) : mlir::Value();
      };
    } else {
      spdlog::error("Unexpected statement type in AST: {}", stmt->toString());
    }
  }
}

std::vector<mlir::Type> MLIRGen::getFunctionArgTypes(FunctionNode *funcNode) {
  std::vector<mlir::Type> argTypes;
  for (const auto &[paramName, paramType] : funcNode->parameters) {
    if (paramType == "image") {
      argTypes.push_back(_builder.getType<ImageType>());
    } else if (paramType == "string") {
      argTypes.push_back(_builder.getType<StringType>());
    } else if (paramType == "f64") {
      argTypes.push_back(_builder.getF64Type());
    } else if (paramType == "int64") {
      argTypes.push_back(_builder.getI64Type());
    } else {
      throw std::runtime_error("Unsupported parameter type: " + paramType);
    }
  }
  return argTypes;
}

void MLIRGen::defineUserFunctions(mlir::ModuleOp module, ModuleNode *root) {
  spdlog::debug("Defining user-defined functions in the MLIR module");
  for (const auto &stmt : root->statements) {
    if (auto funcNode = dynamic_cast<FunctionNode *>(stmt.get())) {
      auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(funcNode->name);
      if (!funcOp) {
        throw std::runtime_error("Function not declared: " + funcNode->name);
      }
      spdlog::debug("Defining function: {}", funcNode->name);

      enterScope(funcNode->name);

      int argIndex = 0;
      for (auto &[paramName, paramType] : funcNode->parameters) {
        auto argValue = funcOp.getArgument(argIndex);
        declareVariable(paramName, argValue);
        ++argIndex;
      }

      mlir::OpBuilder::InsertionGuard guard(_builder);
      _builder.setInsertionPointToStart(funcOp.addEntryBlock());

      for (const auto &bodyStmt : funcNode->body) {
        emitStatement(bodyStmt.get());
      }

      exitScope();

      _builder.create<mlir::func::ReturnOp>(_builder.getUnknownLoc());
    } else {
      spdlog::error("Unexpected statement type in AST: {} -- parsing only FunctionNodes to emit code",
                    stmt->toString());
    }
  }
}

/**
 * @brief Generates MLIR code from the given AST root node.
 * @param root The root node of the AST.
 * @return The generated MLIR module operation.
 */
mlir::ModuleOp MLIRGen::generate(ModuleNode *root) {
  auto module = mlir::ModuleOp::create(_builder.getUnknownLoc());
  registerBuiltinFunctions();

  bool result = normalizeASTMain(module, root);
  if (!result) {
    return nullptr;
  }
  declareUserFunctions(module, root);
  defineUserFunctions(module, root);

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

Result<mlir::Value> MLIRGen::lookupVariable(const std::string &name) const {
  for (auto it = _scopedVariableTable.rbegin(); it != _scopedVariableTable.rend(); ++it) {
    const auto &scope = it->second;
    auto found = scope.find(name);
    if (found != scope.end()) {
      return found->second;
    }
  }
  return std::unexpected(CompileError{"Undefined variable: " + name});
}

void MLIRGen::declareVariable(const std::string &varName, mlir::Value value) {
  if (_scopedVariableTable.empty()) {
    _scopedVariableTable.emplace_back();
  }
  auto &currentScope = _scopedVariableTable.back();
  auto &scopeName = currentScope.first;
  auto &scope = currentScope.second;
  spdlog::debug("Declaring variable: {} for scope: {}", varName, scopeName);
  scope[varName] = value;
}

void MLIRGen::enterScope(const std::string &name) {
  spdlog::debug("Entering scope: {}", name);
  _scopedVariableTable.emplace_back(name, VariableTable());
}

void MLIRGen::exitScope() {
  if (!_scopedVariableTable.empty()) {
    _scopedVariableTable.pop_back();
  } else {
    throw std::runtime_error("Attempted to exit scope when no scopes are active");
  }
}

/**
 * img = load_image("input.png")
 * ^ lhs
 * %0 = load_image("input.png")
 * ^ rhs
 *
 * lhs is the variable name
 * rhs is the SSA value returned from the expression
 */
mlir::Value MLIRGen::emitAssignment(AssignmentNode *node) {
  spdlog::debug("Emitting MLIR for assignment: {}", node->toString());
  auto rhs = emitExpression(node->rhs.get());
  auto varName = node->lhs->name;

  auto var = lookupVariable(varName);
  if (var) {
    throw std::runtime_error("Variable already exists in the current scope and is immutable: " + varName);
  }
  declareVariable(varName, rhs);
  return rhs;
}

mlir::Value MLIRGen::emitCall(CallNode *node) {
  spdlog::debug("Emitting MLIR for call: {}", node->toString());
  std::vector<mlir::Value> args;
  for (auto &arg : node->arguments) {
    args.push_back(emitExpression(arg.get()));
  }
  auto op = emitCallExpression(node, args);
  return op;
}

mlir::Value MLIRGen::emitVariable(VariableNode *node) {
  spdlog::debug("Emitting MLIR for variable: {}", node->toString());
  auto var = lookupVariable(node->name);
  if (var) {
    return var.value();
  } else {
    spdlog::error("lookupVariable failed with error message {}", var.error().message());
    spdlog::error("Undefined variable: {}", node->name);
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

void MLIRGen::registerBuiltinFunctions() {
  _functionTable["load_image"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto imageType = _builder.getType<ImageType>();
    auto filename = args[0];
    auto callOp = _builder.create<LoadImageOp>(loc, imageType, filename);
    return callOp.getResult();
  };
  _functionTable["save_image"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto &filename = args[1];
    _builder.create<SaveImageOp>(loc, inputImage, filename);
    return mlir::Value(); // Return an empty value for void functions
  };
  _functionTable["show_image"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    _builder.create<ShowImageOp>(loc, inputImage);
    return mlir::Value(); // Return an empty value for void functions
  };
  _functionTable["brightness"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto brightnessValue = coerceValueToInt64(_builder, loc, args[1], "brightness", "value");
    auto callOp = _builder.create<BrightnessOp>(loc, inputImage.getType(), inputImage, brightnessValue);
    return callOp.getResult();
  };
  _functionTable["invert"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto callOp = _builder.create<InvertOp>(loc, inputImage.getType(), inputImage);
    return callOp.getResult();
  };
  _functionTable["convolution"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto &kernel = args[1];
    auto callOp = _builder.create<ConvolutionOp>(loc, inputImage.getType(), inputImage, kernel);
    return callOp.getResult();
  };
  _functionTable["sharpen"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto strengthValue = coerceValueToInt64(_builder, loc, args[1], "sharpen", "strength");
    auto callOp = _builder.create<SharpenOp>(loc, inputImage.getType(), inputImage, strengthValue);
    return callOp.getResult();
  };
  _functionTable["box_blur"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto radiusValue = coerceValueToInt64(_builder, loc, args[1], "box_blur", "radius");
    auto callOp = _builder.create<BoxBlurOp>(loc, inputImage.getType(), inputImage, radiusValue);
    return callOp.getResult();
  };
  _functionTable["gaussian_blur"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto radiusValue = coerceValueToInt64(_builder, loc, args[1], "gaussian_blur", "radius");
    auto callOp = _builder.create<GaussianBlurOp>(loc, inputImage.getType(), inputImage, radiusValue);
    return callOp.getResult();
  };
  _functionTable["edge_detect"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto callOp = _builder.create<EdgeDetectOp>(loc, inputImage.getType(), inputImage);
    return callOp.getResult();
  };
  _functionTable["emboss"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto callOp = _builder.create<EmbossOp>(loc, inputImage.getType(), inputImage);
    return callOp.getResult();
  };
  _functionTable["rotate"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto angle = coerceValueToInt64(_builder, loc, args[1], "rotate", "angle");
    auto callOp = _builder.create<RotateOp>(loc, inputImage.getType(), inputImage, angle);
    return callOp.getResult();
  };
  _functionTable["crop"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto x = coerceValueToInt64(_builder, loc, args[1], "crop", "x");
    auto y = coerceValueToInt64(_builder, loc, args[2], "crop", "y");
    auto width = coerceValueToInt64(_builder, loc, args[3], "crop", "width");
    auto height = coerceValueToInt64(_builder, loc, args[4], "crop", "height");
    auto callOp = _builder.create<CropOp>(loc, inputImage.getType(), inputImage, x, y, width, height);
    return callOp.getResult();
  };
  _functionTable["diff"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage1 = args[0];
    auto &inputImage2 = args[1];
    auto callOp = _builder.create<DiffOp>(loc, inputImage1.getType(), inputImage1, inputImage2);
    return callOp.getResult();
  };
  _functionTable["dilate"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto radius = coerceValueToInt64(_builder, loc, args[1], "dilate", "radius");
    auto callOp = _builder.create<DilateOp>(loc, inputImage.getType(), inputImage, radius);
    return callOp.getResult();
  };
  _functionTable["erode"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage = args[0];
    auto radius = coerceValueToInt64(_builder, loc, args[1], "erode", "radius");
    auto callOp = _builder.create<ErodeOp>(loc, inputImage.getType(), inputImage, radius);
    return callOp.getResult();
  };
  _functionTable["blend"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto &inputImage1 = args[0];
    auto &inputImage2 = args[1];
    auto &weight = args[2];
    auto callOp = _builder.create<BlendOp>(loc, inputImage1.getType(), inputImage1, inputImage2, weight);
    return callOp.getResult();
  };
  _functionTable["read_number"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto resultType = _builder.getType<mlir::Float64Type>();
    auto &prompt = args[0];
    auto callOp = _builder.create<ReadNumberOp>(loc, resultType, prompt);
    return callOp.getResult();
  };
  _functionTable["read_string"] = [&](mlir::Location loc, const std::vector<mlir::Value> &args) {
    auto resultType = _builder.getType<StringType>();
    auto &prompt = args[0];
    auto callOp = _builder.create<ReadStringOp>(loc, resultType, prompt);
    return callOp.getResult();
  };
}

mlir::Value MLIRGen::emitCallExpression(CallNode *node, const std::vector<mlir::Value> &args) {
  const auto &name = node->callee;
  mlir::Location loc = _builder.getUnknownLoc();

  auto expectedArgCount = node->arguments.size();
  if (args.size() != expectedArgCount) {
    // TODO: replace with Result<T> and proper error handling, more info on mismatch
    throw std::runtime_error("Function call argument count mismatch for function: " + name);
  }

  if (auto it = _functionTable.find(name); it != _functionTable.end()) {
    auto funcOpResult = it->second(loc, args);
    return funcOpResult;
  } else {
    // TODO: replace with Result<T> and proper error handling, more info
    throw std::runtime_error("Unknown function: " + name);
  }
}

} // namespace picceler
