#include "dialect.h"
#include "ops.h"

namespace picceler {

llvm::LogicalResult KernelConstOp::verify() {
  auto valuesAttr = getValues();
  auto kernel = getResult().getType();

  auto expectedSize = kernel.getRows() * kernel.getCols();
  auto actualSize = valuesAttr.size();

  if (expectedSize != actualSize) {
    return emitOpError("number of values (") << actualSize << ") does not match kernel (" << kernel.getRows() << "x"
                                             << kernel.getCols() << " = " << expectedSize << ")";
  }

  return mlir::success();
}

} // namespace picceler

#define GET_OP_CLASSES
#include "piccelerOps.cpp.inc"
#define GET_OP_CLASSES