#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace picceler {

/**
 * @brief Helper class to generate LLVM IR for accessing fields of the Image
 * struct.
 *
 * Assumes the C++ struct layout:
 * struct Image {
 *   uint32_t _width;      // Offset 0
 *   uint32_t _height;     // Offset 4
 *   unsigned char *_data; // Offset 8
 * };
 */
struct ImageAccessHelper {
  mlir::Value imagePtr;
  mlir::OpBuilder &builder;
  mlir::Location loc;
  mlir::MLIRContext *context;

  ImageAccessHelper(mlir::Value img, mlir::OpBuilder &b, mlir::Location l)
      : imagePtr(img), builder(b), loc(l), context(b.getContext()) {}

  /**
   * @brief Get the LLVM struct type representing the Image struct.
   */
  mlir::Type getImageStructType() {
    mlir::Type i32 = builder.getI32Type();
    mlir::Type ptr = mlir::LLVM::LLVMPointerType::get(context);
    return mlir::LLVM::LLVMStructType::getLiteral(context, {i32, i32, ptr});
  }

  mlir::Value getWidth() {
    auto widthPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(context), getImageStructType(),
        imagePtr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 0});

    return builder.create<mlir::LLVM::LoadOp>(loc, builder.getI32Type(),
                                              widthPtr);
  }

  mlir::Value getHeight() {
    auto heightPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(context), getImageStructType(),
        imagePtr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 1});

    return builder.create<mlir::LLVM::LoadOp>(loc, builder.getI32Type(),
                                              heightPtr);
  }

  mlir::Value getDataPtr() {
    auto dataPtrPtr = builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(context), getImageStructType(),
        imagePtr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});

    return builder.create<mlir::LLVM::LoadOp>(
        loc, mlir::LLVM::LLVMPointerType::get(context), dataPtrPtr);
  }
};

} // namespace picceler