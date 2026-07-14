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
class ImageAccessHelper {
public:
  ImageAccessHelper(mlir::Value ptr, mlir::OpBuilder &b, mlir::Location l);

  /**
   * @brief Defines the logical layout of the C++ struct for GEP offset calculations.
   * struct Image { i32, i32, ptr }
   *
   * @param ctx MLIR context for type creation.
   * @return MLIR type representing the struct layout for GEP indexing.
   */
  static mlir::Type getImageStructType(mlir::MLIRContext *ctx);

  /**
   * @brief Internal helper to create the GEP (Address calculation) for a field index.
   * The expected value of index is 0 for width, 1 for height, and 2 for data pointer.
   * @param index The index of the field for which to calculate the address.
   * @return MLIR value representing the address of the field.
   */
  mlir::Value getFieldAddr(int32_t index);

  /**
   * @brief Gets the value of the width field.
   * @return MLIR value representing the width.
   */
  mlir::Value getWidth();

  /**
   * @brief Gets the value of the height field.
   * @return MLIR value representing the height.
   */
  mlir::Value getHeight();

  /**
   * @brief Gets the value of the data pointer field.
   * @return MLIR value representing the data pointer.
   */
  mlir::Value getDataPtr();

private:
  mlir::Value _structPtr; // !llvm.ptr (Opaque pointer to the Image struct)
  mlir::OpBuilder &_builder;
  mlir::Location _loc;
};

} // namespace picceler
