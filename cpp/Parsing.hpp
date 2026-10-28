#pragma once

#include "Ops.hpp"
#include <llvm/Support/Error.h>
#include <optional>

namespace mlir::inline_ {

/// A position in a text: line and column from 1, the column counting bytes as
/// MLIR's parser does, and the byte offset from the text's start.
struct TextPosition {
  size_t line;
  size_t column;
  size_t byteOffset;
};

/// A failure to parse, inline, or verify a text. `position` is present exactly
/// when the parser reported the failure inside the text.
struct InlineRegionParseError : public llvm::ErrorInfo<InlineRegionParseError> {
  static char ID;

  std::string message;
  std::optional<TextPosition> position;

  inline InlineRegionParseError(std::string message, std::optional<TextPosition> position)
    : message(std::move(message)), position(position) {}

  inline std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  inline void log(llvm::raw_ostream &os) const override {
    os << message;
  }
};

/// Parses `sourceString`, MLIR text in which each operand name is bound to its
/// operand and each type alias name to its type, and appends its ops to
/// `block`. A trailing `inline.yield` is not appended: its operands are the
/// returned values, one for each of `resultTypes`. `loc` is the position of
/// the text's first character in the file the text came from, and each
/// appended op and block argument carries its own position in that file.
llvm::Expected<SmallVector<Value>> parseSourceStringIntoBlock(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    ArrayRef<StringRef> typeAliasNames,
    TypeRange typeAliasTypes,
    TypeRange resultTypes,
    StringRef sourceString,
    Block *block,
    bool verifyAfterParse = true);

}
