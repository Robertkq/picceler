#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace picceler {

/**
 * @brief Helper class to generate LLVM IR for accessing fields of the Image
 * struct via an opaque pointer.
 * Assumes the following C++ struct layout:
 * struct Image {
 *   uint32_t _width;      // Offset 0
 *   uint32_t _height;     // Offset 4
 *   unsigned char *_data; // Offset 8
 * };
 */
struct ImageAccessHelper {
  mlir::Value structPtr; // !llvm.ptr (Opaque pointer to the Image struct)
  mlir::OpBuilder &builder;
  mlir::Location loc;

  ImageAccessHelper(mlir::Value ptr, mlir::OpBuilder &b, mlir::Location l)
      : structPtr(ptr), builder(b), loc(l) {}

  /**
   * Defines the logical layout of the C++ struct for GEP offset calculations.
   * struct Image { i32, i32, ptr }
   */
  static mlir::Type getImageStructType(mlir::MLIRContext *ctx) {
    mlir::Type i32 = mlir::IntegerType::get(ctx, 32);
    mlir::Type ptr = mlir::LLVM::LLVMPointerType::get(ctx);
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, {i32, i32, ptr});
  }

  /**
   * Internal helper to create the GEP (Address calculation) for a field index.
   */
  mlir::Value getFieldAddr(int32_t index) {
    auto structType = getImageStructType(builder.getContext());
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    // We use GEPArg to tell MLIR these are static constant indices.
    // {0, index} means: start at base pointer, go to field 'index'.
    llvm::SmallVector<mlir::LLVM::GEPArg> indices;
    indices.push_back(0);
    indices.push_back(index);

    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        ptrType,    // Result is a pointer to the field
        structType, // The layout we are indexing into
        structPtr,  // The base opaque pointer
        indices);
  }

  mlir::Value getWidth() {
    mlir::Type i32 = mlir::IntegerType::get(builder.getContext(), 32);
    return builder.create<mlir::LLVM::LoadOp>(loc, i32, getFieldAddr(0));
  }

  mlir::Value getHeight() {
    mlir::Type i32 = mlir::IntegerType::get(builder.getContext(), 32);
    return builder.create<mlir::LLVM::LoadOp>(loc, i32, getFieldAddr(1));
  }

  mlir::Value getDataPtr() {
    mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    return builder.create<mlir::LLVM::LoadOp>(loc, ptrType, getFieldAddr(2));
  }
};

} // namespace picceler
