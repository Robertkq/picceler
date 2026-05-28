#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include <utility>

#include "types.h"
#include "piccelerInterfaces.h.inc"

#define GET_OP_CLASSES
#include "piccelerOps.h.inc"
#undef GET_OP_CLASSES

namespace picceler {

/**
 * @brief Common utility function to create a constant float value in MLIR.
 * @param builder The MLIR OpBuilder to use for creating the operation.
 * @param loc The MLIR Location to associate with the operation.
 * @param value The double value to create as a constant.
 * @return An MLIR Value representing the constant float.
 */
mlir::Value createFloatConstant(mlir::OpBuilder &builder, mlir::Location loc, double value);

} // namespace picceler