#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace mlir::inline_ {

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

static LogicalResult invokeAndCaptureDiagnostics(
    MLIRContext *context, 
    std::string& capturedDiagnostics,
    std::function<LogicalResult()> f) {
  capturedDiagnostics.clear();
  llvm::raw_string_ostream os(capturedDiagnostics);

  // create a diagnostic handler that writes to our string
  ScopedDiagnosticHandler handler(context,
    [&os](Diagnostic &diag) {
      diag.print(os);
      os << "\n";
      return success();
    }
  );

  // invoke the function
  LogicalResult result = f();

  os.flush();

  return result;
}

llvm::Expected<InlineRegionOp> parseInlineRegionOpFromSourceString(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    StringRef sourceString) {

  if (operandNames.size() != operands.size()) {
    llvm_unreachable("Internal compiler error: number of operand names doesn't match number of operands");
  }

  // we assume the source string is something like this:
  //
  // inline.inline_region %arg0, %arg1, ... : (arg0_ty, arg1_ty, ...) -> (result_tys...) {
  //   <body>
  // }
  //
  // We can't parse that directly because the parser won't recognize the operand names.
  // So we need to prepend definitions to the source:
  //
  // %arg0 = builtin.unrealized_conversion_cast() { inline_placeholder = 0 } : () -> arg0_ty
  // %arg1 = builtin.unrealized_conversion_cast() { inline_placeholder = 1 } : () -> arg1_ty
  // ...
  // inline.inline_region %arg0, %arg1, ...

  // create a prelude with placeholder values using the provided operand names
  std::string prelude;
  for (size_t i = 0; i < operands.size(); ++i) {
    std::string valueName = operandNames[i].str();
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    operands[i].getType().print(os);

    // create a placeholder SSA value with marker attribute
    prelude += "%" + valueName + " = \"builtin.unrealized_conversion_cast\"() {inline_placeholder = " +
      std::to_string(i) + "} : () -> " + typeStr + "\n";
  }

  // combine prelude with the original source string
  std::string fullStr = prelude + sourceString.str();

  // create a temporary Block and parse these operations into it
  MLIRContext* ctx = loc->getContext();
  Block block;
  std::string diagnostics;
  auto parseResult = invokeAndCaptureDiagnostics(ctx, diagnostics, [&] {
    ParserConfig config(ctx);
    return parseSourceString(fullStr, &block, config);
  });

  if (failed(parseResult)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        diagnostics.empty() ? "Failed to parse source string" : diagnostics
    );
  }

  // replace placeholder values with operands
  for (Operation &op : block) {
    if (auto attr = op.getAttrOfType<IntegerAttr>("inline_placeholder")) {
      // extract the operand index
      int operand_idx = attr.getInt();

      // replace uses with the corresponding block argument
      if (operand_idx >= 0 && operand_idx < operands.size()) {
        op.getResult(0).replaceAllUsesWith(operands[operand_idx]);
      }
    }
  }

  // pick out the InlineRegionOp of interest
  for (Operation &op : block) {
    if (auto inline_region_op = dyn_cast<InlineRegionOp>(op)) {
      // remove the InlineRegionOp from its parent Block
      inline_region_op->remove();
      return inline_region_op;
    }
  }

  llvm_unreachable("Internal compiler error: no InlineRegionOp found in successfully parsed source");
}

} // end mlir::inline_
