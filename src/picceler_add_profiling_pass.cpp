#include "passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include "dialect.h"
#include "ops.h"
#include "types.h"

namespace picceler {

namespace {

/**
 * @brief Ensures a `piccelerTraceBegin`/`piccelerTraceEnd` runtime functions
 * is declared in the module
 */
mlir::func::FuncOp ensureTraceFunc(mlir::ModuleOp module, mlir::OpBuilder &builder, mlir::StringRef name,
                                   mlir::Type stringType) {
  if (auto func = module.lookupSymbol<mlir::func::FuncOp>(name))
    return func;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType({stringType, builder.getI32Type(), builder.getI16Type()}, {});
  auto func = builder.create<mlir::func::FuncOp>(module.getLoc(), name, funcType);
  func.setPrivate();
  return func;
}

} // namespace

#define GEN_PASS_DEF_PICCELERADDPROFILING
#include "piccelerPasses.h.inc"

/**
 * @brief A pass that wraps every Picceler compute/IO op with piccelerTraceBegin/piccelerTraceEnd
 * runtime calls. Excludes StringConstOp (instrumentation itself creates these) and ShowImageOp/
 * ReadNumberOp/ReadStringOp (block on a window/stdin, not compute -- one call would swallow every
 * real op's time into invisibility on the timeline), PrintOp (times stdout buffering, not compute),
 * and KernelConstOp (sub-microsecond next to any real op, just adds a row).
 */
struct PiccelerAddProfilingPass : public impl::PiccelerAddProfilingBase<PiccelerAddProfilingPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    auto stringType = StringType::get(ctx);
    auto beginFunc = ensureTraceFunc(module, builder, "piccelerTraceBegin", stringType);
    auto endFunc = ensureTraceFunc(module, builder, "piccelerTraceEnd", stringType);

    llvm::SmallVector<mlir::Operation *> targets;
    module.walk([&](mlir::Operation *op) {
      if (op->getDialect() && llvm::isa<PiccelerDialect>(op->getDialect()) &&
          !llvm::isa<StringConstOp, ShowImageOp, ReadNumberOp, ReadStringOp, PrintOp, KernelConstOp>(op))
        targets.push_back(op);
    });

    uint32_t opIndex = 0;
    for (mlir::Operation *op : targets) {
      mlir::Location loc = op->getLoc();
      auto nameAttr = builder.getStringAttr(op->getName().getStringRef());

      builder.setInsertionPoint(op);
      auto nameConst = builder.create<StringConstOp>(loc, stringType, nameAttr);
      auto indexConst = builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(opIndex), 32);
      auto trackConst = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 16);
      builder.create<mlir::func::CallOp>(loc, beginFunc, mlir::ValueRange{nameConst, indexConst, trackConst});

      builder.setInsertionPointAfter(op);
      builder.create<mlir::func::CallOp>(loc, endFunc, mlir::ValueRange{nameConst, indexConst, trackConst});

      ++opIndex;
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerAddProfilingPass() { return std::make_unique<PiccelerAddProfilingPass>(); }

} // namespace picceler
