#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "dialect.h"
#include "passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  registry.insert<picceler::PiccelerDialect>();
  picceler::registerPasses();

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}