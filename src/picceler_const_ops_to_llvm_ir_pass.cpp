#include "passes.h"

#include "spdlog/spdlog.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringExtras.h"

#include "ops.h"
#include "types.h"

namespace picceler {

struct StringConstToCall : public mlir::OpRewritePattern<StringConstOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(StringConstOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not inside a ModuleOp");

    auto loc = op.getLoc();
    llvm::StringRef s = op.getValue();
    spdlog::info("StringConst: {}", s);

    // Convert to null-terminated bytes
    std::string bytes(s.begin(), s.end());
    bytes.push_back('\0');

    // Create a unique name for the global
    llvm::SmallString<64> name;
    llvm::hash_code hc = llvm::hash_value(s);
    name += "__picceler_str_";
    name += llvm::utohexstr((uint64_t)hc);

    // LLVM types
    auto *ctx = rewriter.getContext();
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, bytes.size());
    auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);

    // Insert global at the start of the module (only the global, not
    // addressof/GEP)
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      // Create the global string constant
      rewriter.create<mlir::LLVM::GlobalOp>(loc, arrayType, /*isConstant=*/true,
                                            mlir::LLVM::Linkage::Internal, name,
                                            rewriter.getStringAttr(bytes));
    }

    // Now create addressof and GEP at the use site (inside the function, where
    // op lives)
    mlir::Value globalPtr =
        rewriter.create<mlir::LLVM::AddressOfOp>(loc, ptrType, name);

    // Compute GEP to first element (i8* pointer)
    mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));

    mlir::Value elementPtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, arrayType, globalPtr, mlir::ValueRange{zero, zero});

    // Replace the original StringConstOp with this i8* value
    rewriter.replaceOp(op, elementPtr);
    return mlir::success();
  }
};

void PiccelerConstOpsToLLVMIRPass::runOnOperation() {
  spdlog::trace("Running PiccelerConstOpsToLLVMIRPass");
  mlir::ModuleOp module = getOperation();

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<StringConstToCall>(&getContext());

  if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace picceler