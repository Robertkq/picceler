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
 * @brief Ensures a `piccelerTraceBegin`/`piccelerTraceEnd`-shaped runtime function
 * (!picceler.string, i32, i16) -> () is declared in the module, creating it as a private
 * func.func at module scope if it isn't there yet.
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
 * @brief A pass that wraps every Picceler op (except string.const) with piccelerTraceBegin/
 * piccelerTraceEnd runtime calls, so a --profile build produces a Perfetto-viewable trace of what
 * the compiled program actually spent time on. Only added to the pipeline when --profile is
 * passed; see docs/profiling.md.
 *
 * Runs right after canonicalization and before PiccelerFiltersToConvPass, so it sees the ops the
 * user actually wrote (e.g. "picceler.gaussian_blur"), not what they lower into
 * ("picceler.convolution"), and doesn't see dead/folded-away ops the canonicalizer already
 * removed.
 */
struct PiccelerAddProfilingPass : public impl::PiccelerAddProfilingBase<PiccelerAddProfilingPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    auto stringType = StringType::get(ctx);
    auto beginFunc = ensureTraceFunc(module, builder, "piccelerTraceBegin", stringType);
    auto endFunc = ensureTraceFunc(module, builder, "piccelerTraceEnd", stringType);

    // Collect targets before mutating: the loop below creates new picceler.string.const label
    // ops of its own, which must not be picked up and re-instrumented. string.const itself is
    // excluded outright -- it's a compile-time constant materialization (near-zero cost even
    // after lowering), and every excluded literal is one less near-zero-duration noise slice.
    llvm::SmallVector<mlir::Operation *> targets;
    module.walk([&](mlir::Operation *op) {
      if (op->getDialect() && llvm::isa<PiccelerDialect>(op->getDialect()) && !llvm::isa<StringConstOp>(op))
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
      builder.create<mlir::func::CallOp>(loc, beginFunc,
                                         mlir::ValueRange{nameConst, indexConst, trackConst});

      builder.setInsertionPointAfter(op);
      builder.create<mlir::func::CallOp>(loc, endFunc, mlir::ValueRange{nameConst, indexConst, trackConst});

      ++opIndex;
    }
  }
};

std::unique_ptr<mlir::Pass> createPiccelerAddProfilingPass() { return std::make_unique<PiccelerAddProfilingPass>(); }

} // namespace picceler
