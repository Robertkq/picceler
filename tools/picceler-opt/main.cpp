#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "dialect.h"
#include "passes.h"
#include "pass_manager.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register standard dialects (arith, memref, affine, etc.)
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  // Register YOUR Picceler dialect and passes
  registry.insert<picceler::PiccelerDialect>();
  picceler::IRPassManager::registerPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Picceler optimizer driver\n", registry));
}