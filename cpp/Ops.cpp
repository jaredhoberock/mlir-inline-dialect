#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace mlir::inline_ {

ParseResult parseInlineRegionOpSignatureTypes(
    OpAsmParser &parser,
    FunctionType &signatureType) {
  SmallVector<Type> inputTypes;
  SmallVector<Type> resultTypes;

  // parse inputs: must always be wrapped in parenthesis
  if (parser.parseLParen())
    return failure();

  // check if it's an empty input list
  if (parser.parseOptionalRParen().failed()) {
    // it's not empty - parse non-empty input list
    if (parser.parseTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  // optionally parse -> result types
  if (succeeded(parser.parseOptionalArrow())) {
    if (failed(parser.parseOptionalLParen())) {
      Type resultType;
      if (parser.parseType(resultType))
        return failure();
      resultTypes.push_back(resultType);
    } else {
      // parse tuple-style multiple result types
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    }
  }

  signatureType = FunctionType::get(parser.getContext(), inputTypes, resultTypes);
  return success();
}

ParseResult InlineRegionOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands))
    return failure();

  // parse colon
  if (parser.parseColon())
    return failure();

  // parse signature types
  FunctionType funcType;
  if (parseInlineRegionOpSignatureTypes(parser, funcType))
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
  // Print inputs if there are any
  if (!getInputs().empty()) {
    p << " " << getInputs();
  }

  // Print colon and input types in parentheses
  p << " : (";
  llvm::interleaveComma(getInputs().getTypes(), p);
  p << ")";

  // Print result types if nonempty
  if (!getResultTypes().empty()) {
    p << " -> ";
    if (getResultTypes().size() > 1) {
      p << "(";
      llvm::interleaveComma(getResultTypes(), p);
      p << ")";
    } else {
      p << getResultTypes()[0];
    }
  }

  // Ensure region uses operand names
  p << " ";
  p.shadowRegionArgs(getRegion(), getOperands());

  // Print region body (without repeating entry block args)
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);

  // Print optional attributes
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

LogicalResult InlineRegionOp::addParsedRegionBody(ArrayRef<StringRef> inputNames,
                                                  StringRef regionStr) {
  if (inputNames.size() != getInputs().size()) {
    return emitOpError("number of input names does not match expected number of inputs");
  }

  MLIRContext *context = getContext();
  Location loc = getLoc();

  // create a prelude with placeholder values using the provided names
  std::string prelude;
  for (size_t i = 0; i < getInputs().size(); ++i) {
    std::string valueName = inputNames[i].str();
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    getInputs()[i].getType().print(os);

    // create a placeholder with marker attribute
    prelude += "%" + valueName + " = \"builtin.unrealized_conversion_cast\"() {inline_placeholder = " +
      std::to_string(i) + "} : () -> " + typeStr + "\n";
  }

  // combine prelude with the original region string
  std::string fullStr = prelude + regionStr.str();

  // InlineRegionOp::build ensures this block exists
  Block* block = &getRegion().front();

  // parse the combined string into the block
  // don't verify during parsing because YieldOp will complain about its parent
  ParserConfig config(context, /*verifyAfterParse=*/false);
  if (failed(parseSourceString(fullStr, block, config))) {
    return emitOpError("Failed to parse region string");
  }

  // replace placeholder values with block arguments
  SmallVector<Operation*> placeholderOps;
  for (Operation &op : *block) {
    if (auto attr = op.getAttrOfType<IntegerAttr>("inline_placeholder")) {
      // extract the index
      int index = attr.getInt();

      // replace uses with the corresponding block argument
      if (index >= 0 && index < getInputs().size()) {
        op.getResult(0).replaceAllUsesWith(block->getArgument(index));
      }

      // mark for removal
      placeholderOps.push_back(&op);
    }
  }

  // remove the placeholder ops
  for (Operation* op : placeholderOps) {
    op->erase();
  }

  // ensure the block has a terminator
  OpBuilder builder(context);
  ensureTerminator(getRegion(), builder, loc);

  // now verify all child operations once the region is complete
  for (Operation &op : *block) {
    if (failed(mlir::verify(&op)))
      return op.emitError("verification failed after parsing");
  }

  return success();
}

/// Clones the body of an `inline.inline_region` operation using the provided builder.
///
/// The region must contain a single block ending with `inline.yield`. All operations
/// (except the terminator) are cloned before `op`, with block arguments mapped to `inputs`.
/// The mapped yield values are written into `yieldedValues`.
///
/// The original `op` is not erased; the caller is responsible for cleanup.
///
/// Returns `failure()` if the region is malformed (e.g., missing `inline.yield`),
/// in which case `errorMessage` is populated.
LogicalResult InlineRegionOp::cloneBodyAtInsertionPoint(OpBuilder &builder,
                                                        ValueRange inputs,
                                                        SmallVectorImpl<Value> &yieldedValues,
                                                        std::string &errorMessage) {
  // get the region to inline
  Region& sourceRegion = getRegion();
  Block &sourceBlock = sourceRegion.front();

  // check for terminator
  auto yieldOp = dyn_cast<YieldOp>(sourceBlock.getTerminator());
  if (!yieldOp) {
    errorMessage = "expected inline.yield terminator";
    return failure();
  }

  // map block arguments to input values
  IRMapping mapper;
  for (auto [blockArg, operand] : llvm::zip(sourceBlock.getArguments(),
                                            inputs)) {
    mapper.map(blockArg, operand);
  }

  // clone all operations except the terminator
  for (auto& nestedOp : sourceBlock.without_terminator()) {
    builder.clone(nestedOp, mapper);
  }

  // collect the mapped yield values
  yieldedValues.clear();
  for (auto result : yieldOp.getResults()) {
    yieldedValues.push_back(mapper.lookupOrDefault(result));
  }

  return success();
}


} // end mlir::inline_
