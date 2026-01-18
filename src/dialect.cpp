#include "dialect.h"
#include "types.h"
#include "ops.h"
#include "piccelerDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "piccelerTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

namespace picceler {

void PiccelerDialect::initialize() {
  addTypes<ImageType, StringType, KernelType>();
  addOperations<StringConstOp, KernelConstOp, LoadImageOp, ShowImageOp, SaveImageOp, BlurOp, BrightnessOp, InvertOp,
                ConvolutionOp>();
}

} // namespace picceler