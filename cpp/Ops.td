#ifndef INLINE_OPS_TD
#define INLINE_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "Dialect.td"

def InlineRegionOp : InlineOp<"inline_region", [
  AutomaticAllocationScope,
  IsolatedFromAbove,
  OpAsmOpInterface,
  SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Runs a region with explicitly specified arguments and a return value";

  let description = [{
    The `inline.inline_region` operation takes a list of operands and executes a regionj
    with those operands bound to SSA values of the same name within the region.
    referenced by their symbol name within the region.

    Example:
    ```mlir
    %res = inline.inline_region %a, %b : (i32, i32) -> i32 {
      %sum = arith.addi %a, %b : i32
      %two = arith.constant 2 : i32
      %scaled = arith.muli %sum, %two : i32
      yield %scaled
    }
    ```

    This executes the region using values `%a` and `%b`, returning the result of the region.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::TypeRange":$resultTypes)>,
  ];

  let extraClassDeclaration = [{
    mlir::LogicalResult addParsedRegionBody(llvm::ArrayRef<llvm::StringRef> inputNames,
                                            llvm::StringRef regionStr);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "inline"; }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def YieldOp : InlineOp<"yield", [
  HasParent<"InlineRegionOp">,
  Pure,
  ReturnLike,
  Terminator
]> {
  let summary = "Yield operation for inline.inline_region";
  let description = [{
    `inline.yield` terminates an `inline.inline_region` region and returns values to its parent.

    Example:
    ```mlir
    inline.inline_region %a, %b : (i32, i32) -> i32 {
      %sum = arith.addi %a, %b : i32
      inline.yield %sum : i32
    }
    ```
  }];

  // Allow any number of types for operands
  let arguments = (ins Variadic<AnyType>:$results);

  // Simple builder to construct `yield` without explicit operands
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  // Define the assembly format for the `yield` operation
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

#endif // PRINTF_OPS_TD
