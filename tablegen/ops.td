#ifndef PICCELER_OPS_TD
#define PICCELER_OPS_TD

include "mlir/IR/OpBase.td"
include "types.td"

def Picceler_StringConstOp : Op<picceler, "string.const"> {
  let summary = "String constant producing a picceler.string";
  let arguments = (ins StrAttr:$value);
  let results = (outs Picceler_StringType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Picceler_LoadImageOp : Op<picceler, "load_image", []> {
  let summary = "Load an image from a file";
  let arguments = (ins Picceler_StringType:$filename);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_SaveImageOp : Op<picceler, "save_image", []> {
  let summary = "Save an image to a file";
  let arguments = (ins Picceler_ImageType:$input, Picceler_StringType:$filename);
  let results = (outs);
}

def Picceler_ShowImageOp : Op<picceler, "show_image", []> {
  let summary = "Show an image in a window";
  let arguments = (ins Picceler_ImageType:$input);
  let results = (outs);
}

def Picceler_BrightnessOp : Op<picceler, "brightness", []> {
  let summary = "Adjust the brightness of an image";
  let arguments = (ins Picceler_ImageType:$input, I64:$value);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_InvertOp : Op<picceler, "invert", []> {
  let summary = "Invert the pixels of an image";
  let arguments = (ins Picceler_ImageType:$input);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_BlurOp : Op<picceler, "blur", []> {
  let summary = "Apply blur to an image";
  let arguments = (ins Picceler_ImageType:$input, Picceler_StringType:$mode);
  let results = (outs Picceler_ImageType:$result);
}

#endif

