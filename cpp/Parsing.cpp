#include "Parsing.hpp"
#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <llvm/Support/ConvertUTF.h>
#include <llvm/Support/SourceMgr.h>
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


static void adjustErrorLocationToAccountForPrelude(InlineRegionParseError &error,
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


static std::pair<std::string, size_t> buildUccPrelude(
    ArrayRef<StringRef> operandNames,
    ValueRange operands) {
  std::string prelude;
  size_t preludeLineCount = 0;
  for (size_t i = 0; i < operands.size(); ++i) {
    std::string valueName = operandNames[i].str();
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    operands[i].getType().print(os);
    prelude += "%" + valueName +
               " = \"builtin.unrealized_conversion_cast\"() "
               "{inline.placeholder = " + std::to_string(i) + "} : () -> " +
               typeStr + "\n";
    ++preludeLineCount;
  }
  return {prelude, preludeLineCount};
}


static LogicalResult verifyOperationsAndSymbolUses(Block::iterator opsBegin, Block::iterator opsEnd) {
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


llvm::Expected<SmallVector<Value>> parseSourceStringIntoBlock(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    TypeRange resultTypes,
    StringRef sourceString,
    Block *block) {

  MLIRContext *ctx = loc->getContext();

  // build the prelude
  auto [prelude, preludeLines] = buildUccPrelude(operandNames, operands);
  std::string fullSource = prelude + sourceString.str();

  // parse the full source into a temporary block
  Block tempBlock;
  if (auto parseErr = invokeAndCaptureFirstError(ctx, [&] {
        ParserConfig cfg(ctx, /*verifyAfterParse=*/false);
        return parseSourceString(fullSource, &tempBlock, cfg);
      })) {
    // remap diagnostics to the user string to account for the prelude
    adjustErrorLocationToAccountForPrelude(*parseErr, ctx, fullSource,
                                           preludeLines, prelude.size());
    return llvm::make_error<InlineRegionParseError>(*parseErr);
  }

  // collect UCC placeholders
  SmallVector<Value> placeholderValues(operands.size(), Value());
  for (auto op : tempBlock.getOps<UnrealizedConversionCastOp>()) {
    if (auto idx = op->getAttrOfType<IntegerAttr>("inline.placeholder")) {
      int i = idx.getInt();
      if (i >= 0 && static_cast<size_t>(i) < operands.size())
        placeholderValues[i] = op.getResult(0);
    }
  }

  // create a wrapping inline.inline_region op and its body
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(block);
  auto inlineOp = builder.create<InlineRegionOp>(
    loc,
    /*inputs=*/operands,
    /*resultTypes=*/resultTypes
  );

  Region &region = inlineOp.getRegion();
  Block &body = region.front();

  // replace uses of placeholder values with block arguments and erase their defining ops
  for (auto [ph, arg] : llvm::zip(placeholderValues, body.getArguments())) {
    if (ph) {
      ph.replaceAllUsesWith(arg);
      if (Operation *def = ph.getDefiningOp())
        def->erase();
    }
  }

  // move all remaining parsed ops into the region body
  for (Operation &op : llvm::make_early_inc_range(tempBlock)) {
    op.moveBefore(&body, body.end());
  }

  // ensure implicit terminator if missing
  InlineRegionOp::ensureTerminator(inlineOp.getRegion(), builder, loc);

  // inline the body at the end of the destination block
  SmallVector<Value> results;
  std::string errorMsg;
  if (failed(inlineOp.cloneBodyAtInsertionPoint(builder, inlineOp.getInputs(), results, errorMsg)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), errorMsg);

  auto newOpsBegin = std::next(Block::iterator(inlineOp));

  // erase the InlineRegionOp
  inlineOp.erase();

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
