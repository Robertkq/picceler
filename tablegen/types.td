// ===- types.td - Picceler types ------------------------------*- tablegen -*-===//
//
// This file defines the Picceler dialect types.
//
// ===----------------------------------------------------------------------===//

#ifndef PICCELER_TYPES_TD
#define PICCELER_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "dialect.td"

// ===----------------------------------------------------------------------===//
// Image type
// ===----------------------------------------------------------------------===//

def Picceler_ImageType : TypeDef<picceler, "Image"> {
  let mnemonic = "image";
  let summary = "An image object in the Picceler dialect";
  let description = [{
    Represents an abstract image object (pixels, format, etc.)
    manipulated by the Picceler language.
  }];
}

// ===----------------------------------------------------------------------===//
// String type
// ===----------------------------------------------------------------------===//

def Picceler_StringType : TypeDef<picceler, "String"> {
  let mnemonic = "string";
  let summary = "A string value in the Picceler dialect";
  let description = [{
    Represents a runtime string value, e.g., a filename or label.
  }];
}

#endif
