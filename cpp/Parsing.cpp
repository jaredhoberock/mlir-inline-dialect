#include "Parsing.hpp"
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

namespace mlir::inline_ {

char InlineRegionParseError::ID = 0;

// The name of the buffer the parser reads: the prelude, then the caller's text
// from the start of a line.
static constexpr llvm::StringLiteral bufferName = "inline-text";

// The prelude binds each type alias name to its type and each operand name to
// a placeholder of the operand's type, one line each. The placeholders are the
// first ops the buffer parses to, in operand order.
static std::string buildPrelude(ArrayRef<StringRef> operandNames,
                                ValueRange operands,
                                ArrayRef<StringRef> typeAliasNames,
                                TypeRange typeAliasTypes) {
  std::string prelude;
  llvm::raw_string_ostream os(prelude);
  for (auto [name, type] : llvm::zip_equal(typeAliasNames, typeAliasTypes))
    os << "!" << name << " = " << type << "\n";
  for (auto [name, operand] : llvm::zip_equal(operandNames, operands))
    os << "%" << name << " = \"builtin.unrealized_conversion_cast\"() : () -> "
       << operand.getType() << "\n";
  return prelude;
}

// The line and column within the caller's text of `parsed`, a position in the
// parser's buffer, or nullopt when it lies in the prelude's `preludeLines`.
static std::optional<std::pair<unsigned, unsigned>>
textPosition(FileLineColLoc parsed, size_t preludeLines) {
  if (parsed.getLine() <= preludeLines)
    return std::nullopt;
  return std::make_pair(unsigned(parsed.getLine() - preludeLines),
                        parsed.getColumn());
}

// The byte offset into `text` of its (line, column), or nullopt when the text
// has no such position.
static std::optional<size_t> byteOffsetIn(StringRef text, unsigned line,
                                          unsigned column) {
  size_t lineStart = 0;
  for (unsigned i = 1; i < line; ++i) {
    lineStart = text.find('\n', lineStart);
    if (lineStart == StringRef::npos)
      return std::nullopt;
    ++lineStart;
  }
  size_t offset = lineStart + column - 1;
  if (offset > text.size())
    return std::nullopt;
  return offset;
}

// Runs `f` and returns the first error it reports through `context`'s
// diagnostics, as its message and location.
static std::optional<std::pair<std::string, Location>>
firstError(MLIRContext *context, llvm::function_ref<LogicalResult()> f) {
  std::optional<std::pair<std::string, Location>> first;
  ScopedDiagnosticHandler handler(context, [&](Diagnostic &diag) {
    if (!first && diag.getSeverity() == DiagnosticSeverity::Error)
      first.emplace(diag.str(), diag.getLocation());
    return success();
  });
  if (succeeded(f()))
    return std::nullopt;
  return first;
}

static LogicalResult verifyOperationsAndSymbolUses(Block::iterator opsBegin,
                                                   Block::iterator opsEnd) {
  for (Operation &op : llvm::make_range(opsBegin, opsEnd))
    if (failed(verify(&op)))
      return failure();

  SymbolTableCollection symbolTable;
  for (Operation &op : llvm::make_range(opsBegin, opsEnd))
    if (auto symbolUser = dyn_cast<SymbolUserOpInterface>(op))
      if (failed(symbolUser.verifySymbolUses(symbolTable)))
        return failure();

  return success();
}

llvm::Expected<SmallVector<Value>> parseSourceStringIntoBlock(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    ArrayRef<StringRef> typeAliasNames,
    TypeRange typeAliasTypes,
    TypeRange resultTypes,
    StringRef sourceString,
    Block *block,
    bool verifyAfterParse) {
  MLIRContext *ctx = loc->getContext();
  auto failure = [](std::string message,
                    std::optional<TextPosition> position = std::nullopt) {
    return llvm::make_error<InlineRegionParseError>(std::move(message), position);
  };

  std::string prelude = buildPrelude(operandNames, operands, typeAliasNames,
                                     typeAliasTypes);
  size_t preludeLines = llvm::count(prelude, '\n');
  std::string buffer = prelude + sourceString.str();

  Block parsed;
  if (auto error = firstError(ctx, [&] {
        ParserConfig config(ctx, /*verifyAfterParse=*/false);
        return parseSourceString(buffer, &parsed, config, bufferName);
      })) {
    // a parse error reports its position relative to the text
    std::optional<TextPosition> position;
    if (auto at = dyn_cast<FileLineColLoc>(error->second))
      if (auto lineColumn = textPosition(at, preludeLines)) {
        auto [line, column] = *lineColumn;
        if (auto offset = byteOffsetIn(sourceString, line, column))
          position = TextPosition{line, column, *offset};
      }
    return failure(error->first, position);
  }

  // every op and block argument the parser placed in its buffer takes its
  // position in the caller's file: the text's first line continues `loc`'s
  // line, and each later text line starts at column 1 of a following line,
  // which holds when the text appears there verbatim
  AttrTypeReplacer relocator;
  relocator.addReplacement([&](FileLineColLoc at) -> std::optional<Attribute> {
    if (at.getFilename() != bufferName)
      return std::nullopt;
    auto start = dyn_cast<FileLineColRange>(loc);
    auto lineColumn = textPosition(at, preludeLines);
    if (!start || !lineColumn)
      return loc;
    auto [line, column] = *lineColumn;
    return FileLineColLoc::get(
        start.getFilename(), start.getStartLine() + line - 1,
        line == 1 ? column + start.getStartColumn() - 1 : column);
  });
  for (Operation &op : parsed)
    relocator.recursivelyReplaceElementsIn(&op, /*replaceAttrs=*/false,
                                           /*replaceLocs=*/true);

  // the placeholders stand for the operands
  for (Value operand : operands) {
    Operation &placeholder = parsed.front();
    placeholder.getResult(0).replaceAllUsesWith(operand);
    placeholder.erase();
  }

  // a trailing terminator is the text's inline.yield, whose operands are the
  // text's results
  SmallVector<Value> results;
  if (!parsed.empty() && parsed.back().hasTrait<OpTrait::IsTerminator>()) {
    auto yield = dyn_cast<YieldOp>(parsed.back());
    if (!yield)
      return failure("expected inline.yield terminator");
    results.assign(yield.getResults().begin(), yield.getResults().end());
    yield.erase();
  }
  if (results.size() != resultTypes.size())
    return failure(("inline asm yields " + Twine(results.size()) +
                    " values but " + Twine(resultTypes.size()) + " expected").str());

  Operation *firstNew = parsed.empty() ? nullptr : &parsed.front();
  block->getOperations().splice(block->end(), parsed.getOperations());

  if (verifyAfterParse) {
    Block::iterator newOps = firstNew ? Block::iterator(firstNew) : block->end();
    if (auto error = firstError(ctx, [&] {
          return verifyOperationsAndSymbolUses(newOps, block->end());
        }))
      return failure(error->first);
  }

  return results;
}

}
