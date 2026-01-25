# Dialect
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/dialect.td)
mlir_tablegen(piccelerDialect.h.inc -gen-dialect-decls -dialect=picceler -I${CMAKE_CURRENT_SOURCE_DIR})
mlir_tablegen(piccelerDialect.cpp.inc -gen-dialect-defs -dialect=picceler -I${CMAKE_CURRENT_SOURCE_DIR})
add_public_tablegen_target(PiccelerDialectIncGen)

# Types
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/types.td)
mlir_tablegen(piccelerTypes.h.inc -gen-typedef-decls -dialect=picceler -I${CMAKE_CURRENT_SOURCE_DIR})
mlir_tablegen(piccelerTypes.cpp.inc -gen-typedef-defs -dialect=picceler -I${CMAKE_CURRENT_SOURCE_DIR})
add_public_tablegen_target(PiccelerTypesIncGen)

# Ops
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/ops.td)
mlir_tablegen(piccelerOps.h.inc -gen-op-decls -dialect=picceler -I${CMAKE_CURRENT_SOURCE_DIR})
mlir_tablegen(piccelerOps.cpp.inc -gen-op-defs -dialect=picceler -I${CMAKE_CURRENT_SOURCE_DIR})
add_public_tablegen_target(PiccelerOpsIncGen)

# Passes
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/passes.td)
mlir_tablegen(piccelerPasses.h.inc -gen-pass-decls -I${CMAKE_CURRENT_SOURCE_DIR})
add_public_tablegen_target(PiccelerPassesIncGen)

set(PICCELER_TABLEGEN_TARGETS
  PiccelerDialectIncGen
  PiccelerTypesIncGen
  PiccelerOpsIncGen
  PiccelerPassesIncGen
  PARENT_SCOPE
)