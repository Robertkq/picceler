#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "types.h"

#define GET_OP_CLASSES
#include "piccelerOps.h.inc"
#undef GET_OP_CLASSES
