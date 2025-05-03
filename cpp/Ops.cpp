#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/Builders.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::inline_;

ParseResult InlineRegionOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands))
    return failure();

  // parse colon and input/output types
  FunctionType funcType;
  if (parser.parseColonType(funcType))
    return failure();

  // resolve operands against function input types
  if (parser.resolveOperands(operands, funcType.getInputs(), parser.getNameLoc(), result.operands))
    return failure();

  // add result types
  result.types.append(funcType.getResults().begin(), funcType.getResults().end());

  // construct argument list for block
  SmallVector<OpAsmParser::Argument> args;
  for (size_t i = 0; i < operands.size(); ++i) {
    OpAsmParser::Argument arg;
    arg.ssaName = operands[i];
    arg.type = funcType.getInput(i);
    args.push_back(arg);
  }

  // parse region with typed args
  if (parser.parseRegion(*result.addRegion(), args, /*enableNameShadowing=*/true))
    return failure();

  ensureTerminator(*result.regions.front(), parser.getBuilder(), result.location);

  return success();
}

void InlineRegionOp::print(OpAsmPrinter &p) {
  p << " " << getInputs();
  p << " : ";
  p.printFunctionalType(getInputs().getTypes(), getResultTypes());
  p << " ";
  p.shadowRegionArgs(getRegion(), getOperands());
  p.printRegion(getRegion(), /* printEntryBlockArgs = */ false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult InlineRegionOp::verify() {
  Block &block = getRegion().front();
  
  // Verify block arguments match input types
  if (block.getNumArguments() != getInputs().size())
    return emitOpError("expected block to have same number of arguments as inputs");
    
  for (auto [blockArg, input]: llvm::zip(block.getArguments(), getInputs())) {
    if (blockArg.getType() != input.getType())
      return emitOpError("block argument type ")
             << blockArg.getType() 
             << " does not match input type "
             << input.getType();
  }

  if (block.empty())
    return emitOpError("region block must not be empty");
  
  // Get the yield op
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getOperands().size() != getResults().size())
    return emitOpError("expected same number of yield values and results");

  // For each result, verify the yield type matches the unwrapped result type
  for (auto [yieldOperand, result] : llvm::zip(yieldOp.getOperands(), getResults())) {
    Type yieldType = yieldOperand.getType();
    if (yieldType != result.getType()) {
      return emitOpError("yield type ")
             << yieldType
             << " doesn't match result type "
             << result.getType();
    }
  }

  return success();
}

void InlineRegionOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange inputs, TypeRange resultTypes) {
  result.addOperands(inputs);
  result.addTypes(resultTypes);
  Region *bodyRegion = result.addRegion();
  Block *body = new Block();

  // add block arguments
  for (Value input : inputs) {
    body->addArgument(input.getType(), builder.getUnknownLoc());
  }

  bodyRegion->push_back(body);
}
