#include "ops.h"

#define GET_OP_INTERFACE_DEFS
#include "piccelerInterfaces.cpp.inc"
#undef GET_OP_INTERFACE_DEFS

#define GET_OP_CLASSES
#include "piccelerOps.cpp.inc"
#undef GET_OP_CLASSES