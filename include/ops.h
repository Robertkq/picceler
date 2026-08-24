#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <utility>

#include "types.h"
#include "error.h"
#include "channels.h"
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

/**
 * @brief Common utility function to create a constant int value in MLIR.
 * @param builder The MLIR OpBuilder to use for creating the operation.
 * @param loc The MLIR Location to associate with the operation.
 * @param value The int64_t value to create as a constant.
 * @return An MLIR Value representing the constant int.
 */
mlir::Value createIntConstant(mlir::OpBuilder &builder, mlir::Location loc, int64_t value);

/**
 * @brief Builds an `affine.parallel` band with one induction variable per entry in `upperBounds`, lower bound 0 and
 * step 1. Upper bounds may be dynamic SSA values (e.g. a runtime-computed neighborhood/kernel size), not just
 * compile-time constants. Pass `resultTypes`/`reductions` to get a reduction band whose body must end with a matching
 * `affine.yield`; left empty, the trivial `affine.yield` terminator is inserted automatically.
 *
 * Shared between `PiccelerToAffinePass` (pixel/neighborhood loops) and `PiccelerFiltersToConvPass` (runtime kernel
 * fill loops for box_blur/gaussian_blur with a non-constant radius).
 *
 * @param rewriter The rewriter to use for creating the affine.parallel op and its bounds.
 * @param loc The location to associate with the created ops.
 * @param upperBounds One SSA value per loop dimension, used as the (exclusive) upper bound.
 * @param resultTypes Result types for a reduction band, or empty for a plain loop.
 * @param reductions One reduction kind per result, matching resultTypes.
 * @return The created `affine.parallel` op.
 */
mlir::affine::AffineParallelOp createAffineParallel(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                                                    mlir::ValueRange upperBounds, mlir::TypeRange resultTypes = {},
                                                    llvm::ArrayRef<mlir::arith::AtomicRMWKind> reductions = {});

} // namespace picceler