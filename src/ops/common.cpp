#include "mlir/Dialect/Arith/IR/Arith.h"

namespace picceler {

mlir::Value createFloatConstant(mlir::OpBuilder &builder, mlir::Location loc, double value) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, builder.getF64Type(), llvm::APFloat(value));
}

} // namespace picceler