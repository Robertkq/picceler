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

  ImageAccessHelper(mlir::Value ptr, mlir::OpBuilder &b, mlir::Location l);

  /**
   * Defines the logical layout of the C++ struct for GEP offset calculations.
   * struct Image { i32, i32, ptr }
   */
  static mlir::Type getImageStructType(mlir::MLIRContext *ctx);

  /**
   * Internal helper to create the GEP (Address calculation) for a field index.
   */
  mlir::Value getFieldAddr(int32_t index);

  mlir::Value getWidth();

  mlir::Value getHeight();

  mlir::Value getDataPtr();
};

} // namespace picceler
