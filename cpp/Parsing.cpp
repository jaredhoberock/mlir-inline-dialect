#include "Parsing.hpp"
#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/ConvertUTF.h>
#include <mlir/Parser/Parser.h>

namespace mlir::inline_ {

char InlineRegionParseError::ID = 0;

static std::optional<InlineRegionParseError> invokeAndCaptureFirstError(
    MLIRContext *context, 
    std::function<LogicalResult()> f) {

  std::optional<InlineRegionParseError> captured;

  // create a diagnostic handler that writes to our string
  ScopedDiagnosticHandler handler(context, [&](Diagnostic &diag) {
    // only capture first error and ignore things that aren't errors
    if (captured || diag.getSeverity() != DiagnosticSeverity::Error)
    if (captured.has_value())
      return success();

    std::string message;
    llvm::raw_string_ostream os(message);
    os << diag.str(); // this omits the location prefix
    os.flush();

    captured = InlineRegionParseError(message, diag.getLocation(), std::nullopt);
    return success();
  });

  // invoke the function and ignore the result
  // we assume that if f produced failure(), then captured has an error in it
  if (failed(f())) {
    // presumably captured has an error inside it
    // assert(captured.has_value());
  }

  return captured;
}

static std::optional<size_t> findByteOffsetOfLoc(llvm::StringRef buffer, unsigned targetLine, unsigned targetCol) {
  if (targetLine == 0 || targetCol == 0)
    return std::nullopt;

  size_t byteOffset = 0;
  unsigned curLine = 1;

  // Move to start of target line
  while (curLine < targetLine) {
    size_t newlinePos = buffer.find('\n');
    if (newlinePos == llvm::StringRef::npos)
      return std::nullopt;
    byteOffset += newlinePos + 1;
    buffer = buffer.drop_front(newlinePos + 1);
    ++curLine;
  }

  // Now buffer starts at target line; walk characters to compute byte offset of column
  llvm::StringRef lineText = buffer.take_until([](char c) { return c == '\n'; });
  size_t colOffset = 0;
  unsigned charCount = 0;

  for (auto it = lineText.begin(); it != lineText.end(); ) {
    if (charCount == targetCol - 1)
      break;
    unsigned charLen = llvm::getNumBytesForUTF8(*it);
    if (charLen == 0 || std::distance(it, lineText.end()) < charLen)
      return std::nullopt;
    it += charLen;
    colOffset += charLen;
    ++charCount;
  }

  return byteOffset + colOffset;
}


void adjustErrorLocationToAccountForPrelude(InlineRegionParseError &error,
                                            MLIRContext* ctx,
                                            StringRef fullSourceStr,
                                            size_t numPreludeLines,
                                            size_t numPreludeBytes) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(error.loc)) {
    unsigned line = fileLoc.getLine();
    unsigned col = fileLoc.getColumn();
    auto byteOffset = findByteOffsetOfLoc(fullSourceStr, line, col);
  
    // adjust line, col, & byteOffset to account for the prelude
    if (line > numPreludeLines) {
      // location is within the user's code
      unsigned adjustedLine = line - numPreludeLines;
      error.loc = FileLineColLoc::get(ctx, fileLoc.getFilename(), adjustedLine, col);
    } else {
      // location is in the prelude, collapse to start of user region
      error.loc = FileLineColLoc::get(ctx, fileLoc.getFilename(), 1, 1);
      byteOffset = 0;
    }
  
    // adjust byte offset if it was found
    if (byteOffset && *byteOffset >= numPreludeBytes) {
      // location is within the user's code
      error.byteOffset = *byteOffset - numPreludeBytes;
    } else {
      // location is within the prelude
      error.byteOffset = std::nullopt;
    }
  }
}


llvm::Expected<InlineRegionOp> parseInlineRegionOpFromSourceString(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    StringRef sourceString,
    bool verifyAfterParse) {

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
  size_t preludeLineCount = 0;
  for (size_t i = 0; i < operands.size(); ++i) {
    std::string valueName = operandNames[i].str();
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    operands[i].getType().print(os);

    // create a placeholder SSA value with marker attribute
    prelude += "%" + valueName + " = \"builtin.unrealized_conversion_cast\"() {inline_placeholder = " +
      std::to_string(i) + "} : () -> " + typeStr + "\n";
    ++preludeLineCount;
  }

  // combine prelude with the original source string
  std::string fullStr = prelude + sourceString.str();

  // create a temporary Block and parse these operations into it
  MLIRContext* ctx = loc->getContext();
  Block block;
  auto error = invokeAndCaptureFirstError(ctx, [&] {
    ParserConfig config(ctx, verifyAfterParse);
    return parseSourceString(fullStr, &block, config);
  });

  // if there was an error, adjust the error location
  // to account for the prelude
  if (error) {
    adjustErrorLocationToAccountForPrelude(*error,
                                           ctx,
                                           fullStr,
                                           preludeLineCount, prelude.size());
    return llvm::make_error<InlineRegionParseError>(*error);
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


LogicalResult verifyOperationsAndSymbolUses(Block::iterator opsBegin, Block::iterator opsEnd) {
  // first verify each operation individually
  for (auto op = opsBegin; op != opsEnd; ++op) {
    if (failed(verify(&*op))) {
      return failure();
    }
  }
  
  // now verify symbol uses across all operations
  SymbolTableCollection symbolTable;
  for (auto op = opsBegin; op != opsEnd; ++op) {
    if (auto symbolUser = dyn_cast<SymbolUserOpInterface>(*op)) {
      if (failed(symbolUser.verifySymbolUses(symbolTable))) {
        return failure();
      }
    }
  }

  return success();
}


// `%arg0, %arg1, ...`
std::string buildOperandNamesString(ArrayRef<StringRef> names) {
  // prepend '%' to every name
  SmallVector<std::string> prefixed;
  prefixed.reserve(names.size());
  for (StringRef name : names)
    prefixed.push_back(("%" + name).str());

  return join(prefixed, ", ");
}


// `arg_ty0, arg_ty1, ...`
std::string buildOperandTypesString(ValueRange values) {
  std::string out;
  llvm::raw_string_ostream os(out);

  bool first = true;
  for (Value v : values) {
    if (!first)
      os << ", ";
    first = false;

    v.getType().print(os);
  }
  os.flush();
  return out;
}


std::string buildResultTypesString(TypeRange types) {
  // if there are no results, return the empty string
  if (types.empty())
    return {};

  std::string out;
  llvm::raw_string_ostream os(out);

  os << "-> ";

  if (types.size() == 1) {
    // if there is a single result, return "-> result_ty0"
    types[0].print(os);
  } else {
    // otherwise, wrap in parens
    os << "(";

    bool first = true;
    for (Type ty : types) {
      if (!first)
        os << ", ";
      first = false;

      ty.print(os);
    }

    os << ")";
  }
  os.flush();
  return out;
}


std::pair<std::string,std::string> buildInlineRegionSourceStringSuffixAndPrefix(
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    TypeRange resultTypes) {
  std::string operandNamesStr = buildOperandNamesString(operandNames);
  std::string operandTypesStr = buildOperandTypesString(operands);
  std::string resultTypesStr = buildResultTypesString(resultTypes);

  std::string prefixStr = "inline.inline_region " + operandNamesStr + " : (" + operandTypesStr + ") " + resultTypesStr + " { ";
  std::string suffixStr = "}";

  return std::make_pair(prefixStr, suffixStr);
}


llvm::Expected<SmallVector<Value>> parseSourceStringIntoBlock(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    TypeRange resultTypes,
    StringRef sourceString,
    Block *block) {

  // wrap the source string into a full inline.inline_region operation
  std::string prefixString, suffixString;
  std::tie(prefixString, suffixString)
    = buildInlineRegionSourceStringSuffixAndPrefix(operandNames, operands, resultTypes);

  std::string fullSourceString = prefixString + "\n" + std::string(sourceString) + "\n" + suffixString;

  // parse the inline.inline_region op
  // don't verify the ops while parsing. we'll do that below once they're
  // in the destination block
  llvm::Expected<InlineRegionOp> maybeOp =
    parseInlineRegionOpFromSourceString(loc, operandNames, operands,
                                        fullSourceString, /*verifyAfterParse=*/false);

  // if there was an error, adjust error locations so that they point into the sourceString
  // that was actually passed by the caller
  if (!maybeOp) {
    return llvm::handleErrors(maybeOp.takeError(), [&](InlineRegionParseError &error) {
      adjustErrorLocationToAccountForPrelude(error,
                                             loc.getContext(),
                                             fullSourceString,
                                             1, // the prefix is constructed to be exactly one line
                                             prefixString.size());
      return llvm::make_error<InlineRegionParseError>(std::move(error));
    });
  }

  auto op = *maybeOp;

  // insert the op at the end of the block
  block->push_back(op);

  // inline the body at the end of the block
  OpBuilder builder(op->getContext());
  builder.setInsertionPointToEnd(block);

  SmallVector<Value> results;
  std::string error;
  if (failed(op.cloneBodyAtInsertionPoint(builder, op.getInputs(), results, error)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), error);

  auto newOpsBegin = std::next(Block::iterator(op));

  // erase the InlineRegionOp
  op.erase();

  // verify the new ops
  auto verifyError = invokeAndCaptureFirstError(loc->getContext(), [&] {
    return verifyOperationsAndSymbolUses(newOpsBegin, block->end());
  });

  // check for a verification error
  if (verifyError)
    return llvm::make_error<InlineRegionParseError>(*verifyError);

  // success; return the results
  return results;
}


}
