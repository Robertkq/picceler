#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

class PiccelerDialect : public mlir::Dialect {
public:
  explicit PiccelerDialect(mlir::MLIRContext *context);

  static llvm::StringRef getDialectNamespace() { return "picceler"; }
};