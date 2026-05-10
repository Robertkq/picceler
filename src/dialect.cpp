#include "dialect.h"
#include "types.h"
#include "ops.h"
#include "piccelerDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "piccelerTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

namespace picceler {

void PiccelerDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "piccelerTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
      >();
  addOperations<
#define GET_OP_LIST
#include "piccelerOps.cpp.inc"
#undef GET_OP_LIST
      >();
}

} // namespace picceler
