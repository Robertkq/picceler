#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "dialect.h"
#include "passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<picceler::PiccelerDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect>();

  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  picceler::registerPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Picceler optimizer driver\n", registry));
}