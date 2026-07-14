#include "image_access_helper.h"

namespace picceler {
ImageAccessHelper::ImageAccessHelper(mlir::Value ptr, mlir::OpBuilder &b, mlir::Location l)
    : _structPtr(ptr), _builder(b), _loc(l) {}

mlir::Type ImageAccessHelper::getImageStructType(mlir::MLIRContext *ctx) {
  mlir::Type i32 = mlir::IntegerType::get(ctx, 32);
  mlir::Type ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, {i32, i32, ptr});
}

mlir::Value ImageAccessHelper::getFieldAddr(int32_t index) {
  auto structType = getImageStructType(_builder.getContext());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(_builder.getContext());

  // We use GEPArg to tell MLIR these are static constant indices.
  // {0, index} means: start at base pointer, go to field 'index'.
  llvm::SmallVector<mlir::LLVM::GEPArg> indices;
  indices.push_back(0);
  indices.push_back(index);

  return _builder.create<mlir::LLVM::GEPOp>(_loc,
                                            ptrType,    // Result is a pointer to the field
                                            structType, // The layout we are indexing into
                                            _structPtr, // The base opaque pointer
                                            indices);
}

mlir::Value ImageAccessHelper::getWidth() {
  mlir::Type i32 = mlir::IntegerType::get(_builder.getContext(), 32);
  return _builder.create<mlir::LLVM::LoadOp>(_loc, i32, getFieldAddr(0));
}

mlir::Value ImageAccessHelper::getHeight() {
  mlir::Type i32 = mlir::IntegerType::get(_builder.getContext(), 32);
  return _builder.create<mlir::LLVM::LoadOp>(_loc, i32, getFieldAddr(1));
}

mlir::Value ImageAccessHelper::getDataPtr() {
  mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(_builder.getContext());
  return _builder.create<mlir::LLVM::LoadOp>(_loc, ptrType, getFieldAddr(2));
}
} // namespace picceler