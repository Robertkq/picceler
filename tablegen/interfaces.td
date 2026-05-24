// ===- interfaces.td - Picceler op interfaces -------------*- tablegen -*-===//
//
// This file defines shared interfaces for Picceler dialect ops.
//
// ===----------------------------------------------------------------------===//

#ifndef PICCELER_INTERFACES_TD
#define PICCELER_INTERFACES_TD

include "mlir/IR/OpBase.td"

def NeighbourhoodOpInterface : OpInterface<"NeighbourhoodOpInterface"> {
	let description = [{
		Interface for ops that lower through a neighborhood or sliding-window
		reduction.
	}];

	let methods = [
		InterfaceMethod<
			"Returns the initial accumulator value for the reduction.",
			"mlir::Value",
			"initializeAccumulator",
			(ins "mlir::OpBuilder &":$builder, "mlir::Location":$loc)
		>,
		InterfaceMethod<
			"Updates the accumulator with one pixel sample and an optional kernel value.",
			"mlir::Value",
			"accumulate",
			(ins "mlir::OpBuilder &":$builder, "mlir::Location":$loc,
					 "mlir::Value":$currentAcc, "mlir::Value":$pixelValue,
					 "mlir::Value":$optionalKernelValue)
		>,
		InterfaceMethod<
			"Performs any final adjustment before the accumulated result is written out.",
			"mlir::Value",
			"finalizeAccumulator",
			(ins "mlir::OpBuilder &":$builder, "mlir::Location":$loc,
					 "mlir::Value":$finalAcc)
		>,
        InterfaceMethod<
		"Returns the neighborhood shape to be processed around each pixel.",
		"std::pair<uint64_t, uint64_t>",
        "getNeighborhoodSize",
        (ins "mlir::ArrayRef<mlir::Value>":$operands)
        >
	];
}

def ElementWiseUnaryOpInterface : OpInterface<"ElementWiseUnaryOpInterface"> {
  let methods = [
    InterfaceMethod<
      "Transforms one input pixel into one output pixel.",
      "mlir::Value",
      "transformPixel",
      (ins "mlir::OpBuilder &":$builder,
           "mlir::Location":$loc,
           "mlir::Value":$inputPixel)
    >
  ];
}

def ElementWiseBinaryOpInterface : OpInterface<"ElementWiseBinaryOpInterface"> {
  let methods = [
    InterfaceMethod<
      "Transforms two input pixels into one output pixel.",
      "mlir::Value",
      "transformPixels",
      (ins "mlir::OpBuilder &":$builder,
           "mlir::Location":$loc,
           "mlir::Value":$lhsPixel,
           "mlir::Value":$rhsPixel)
    >
  ];
}

#endif
