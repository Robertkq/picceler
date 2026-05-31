#include "mlir/Dialect/Arith/IR/Arith.h"

namespace picceler {

mlir::Value createFloatConstant(mlir::OpBuilder &builder, mlir::Location loc, double value) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, builder.getF64Type(), llvm::APFloat(value));
}

mlir::Value createIntConstant(mlir::OpBuilder &builder, mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64);
}

} // namespace picceler