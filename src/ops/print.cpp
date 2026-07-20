#include "ops.h"

#include "spdlog/spdlog.h"

namespace picceler {

mlir::LogicalResult PrintOp::verify() {
  auto fmt = getFmt();
  auto producer = fmt.getDefiningOp();
  if (!producer || !llvm::isa<StringConstOp>(producer)) {
    spdlog::error("fmt operand of print must be compile time constant");
    return mlir::failure();
  }
  return mlir::success();
}
} // namespace picceler