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

def Picceler_ImageType : TypeDef<PiccelerDialect, "Image"> {
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

def Picceler_StringType : TypeDef<PiccelerDialect, "String"> {
  let mnemonic = "string";
  let summary = "A string value in the Picceler dialect";
  let description = [{
    Represents a runtime string value, e.g., a filename or label.
  }];
}

// ===----------------------------------------------------------------------===//
// Kernel type
// ===----------------------------------------------------------------------===//

def Picceler_KernelType : TypeDef<PiccelerDialect, "Kernel"> {
  let mnemonic = "kernel";
  let summary = "A 2D matrix kernel for convolution";
  let description = [{
    Represents a 2D convolution kernel used in image processing.
  }];
  let parameters = (ins "int":$rows, "int":$cols);
  let assemblyFormat = "`<` $rows `x` $cols `>`";
}

#endif
