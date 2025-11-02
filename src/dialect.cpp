#include "dialect.h"
#include "types.h"
#include "ops.h"
#include "piccelerDialect.cpp.inc"
#include "piccelerTypes.cpp.inc"

namespace picceler {

void picceler::initialize() {
  addTypes<ImageType, StringType>();
  addOperations<StringConstOp, LoadImageOp, ShowImageOp, SaveImageOp, BlurOp>();
}

} // namespace picceler