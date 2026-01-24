#ifndef PICCELER_OPS_TD
#define PICCELER_OPS_TD

include "mlir/IR/OpBase.td"
include "types.td"

def Picceler_StringConstOp : Op<PiccelerDialect, "string.const"> {
  let summary = "String constant producing a picceler.string";
  let arguments = (ins StrAttr:$value);
  let results = (outs Picceler_StringType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Picceler_KernelConstOp : Op<PiccelerDialect, "kernel.const"> {
  let summary = "Kernel constant producing a picceler.kernel";
  let arguments = (ins F64ElementsAttr:$values);
  let results = (outs Picceler_KernelType:$result);
  let hasVerifier = 1;
}

def Picceler_LoadImageOp : Op<PiccelerDialect, "load_image", []> {
  let summary = "Load an image from a file";
  let arguments = (ins Picceler_StringType:$filename);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_SaveImageOp : Op<PiccelerDialect, "save_image", []> {
  let summary = "Save an image to a file";
  let arguments = (ins Picceler_ImageType:$input, Picceler_StringType:$filename);
  let results = (outs);
}

def Picceler_ShowImageOp : Op<PiccelerDialect, "show_image", []> {
  let summary = "Show an image in a window";
  let arguments = (ins Picceler_ImageType:$input);
  let results = (outs);
}

def Picceler_BrightnessOp : Op<PiccelerDialect, "brightness", []> {
  let summary = "Adjust the brightness of an image";
  let arguments = (ins Picceler_ImageType:$input, I64:$value);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_InvertOp : Op<PiccelerDialect, "invert", []> {
  let summary = "Invert the pixels of an image";
  let arguments = (ins Picceler_ImageType:$input);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_ConvolutionOp : Op<PiccelerDialect, "convolution", []> {
  let summary = "Apply convolution to an image with a kernel";
  let arguments = (ins Picceler_ImageType:$input, AnyTypeOf<[Picceler_KernelType, AnyMemRef]>:$kernel);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_BlurOp : Op<PiccelerDialect, "blur", []> {
  let summary = "Apply blur to an image";
  let arguments = (ins Picceler_ImageType:$input, Picceler_StringType:$mode);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_SharpenOp : Op<PiccelerDialect, "sharpen", []> {
  let summary = "Adjust the sharpness of an image";
  let arguments = (ins Picceler_ImageType:$input, I64:$value);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_BoxBlurOp : Op<PiccelerDialect, "box_blur", []> {
  let summary = "Apply a simple box blur";
  let arguments = (ins Picceler_ImageType:$input, I64:$radius);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_GaussianBlurOp : Op<PiccelerDialect, "gaussian_blur", []> {
  let summary = "Apply a gaussian blur (smoother)";
  let arguments = (ins Picceler_ImageType:$input, I64:$radius);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_EdgeDetectOp : Op<PiccelerDialect, "edge_detect", []> {
  let summary = "Detect edges in the image";
  let arguments = (ins Picceler_ImageType:$input);
  let results = (outs Picceler_ImageType:$result);
}

def Picceler_EmbossOp : Op<PiccelerDialect, "emboss", []> {
  let summary = "Apply a 3D emboss effect";
  let arguments = (ins Picceler_ImageType:$input);
  let results = (outs Picceler_ImageType:$result);
}

#endif

