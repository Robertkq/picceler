// ===- dialect.td - Dialect specification ---------------------*- tablegen -*-===//
//
// This file defines the Picceler dialect.
//
// ===----------------------------------------------------------------------===//

#ifndef PICCELER_DIALECT_TD
#define PICCELER_DIALECT_TD

include "mlir/IR/DialectBase.td"

def picceler : Dialect {
    let summary = "Main dialect for the Picceler language";
    let description = [{
        This dialect defines the core operations and types for the Picceler language.
    }];
    let name = "picceler";
    let cppNamespace = "::picceler";
    let useDefaultTypePrinterParser = 1;
}

#endif

