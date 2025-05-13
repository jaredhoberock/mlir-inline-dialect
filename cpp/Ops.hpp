#pragma once

#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <optional>
#include <tuple>

#define GET_OP_CLASSES
#include "Ops.hpp.inc"

namespace mlir::inline_ {

struct InlineRegionParseError : public llvm::ErrorInfo<InlineRegionParseError> {
  static char ID;

  std::string message;
  Location loc;
  std::optional<size_t> byteOffset;

  inline InlineRegionParseError(std::string message, Location loc, std::optional<size_t> offset)
    : message(message), loc(loc), byteOffset(offset) {}

  inline std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  inline void log(llvm::raw_ostream &os) const override {
    os << message;
  }

  // returns the source (line, column, byteOffset) of the parse error
  // *only* if all three items are available
  inline std::optional<std::tuple<size_t,size_t,size_t>> getLocation() const {
    if (!byteOffset.has_value())
      return std::nullopt;

    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      return std::make_tuple(
        fileLoc.getLine(),
        fileLoc.getColumn(),
        *byteOffset);
    }

    return std::nullopt;
  }
};

llvm::Expected<InlineRegionOp> parseInlineRegionOpFromSourceString(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    StringRef sourceString,
    bool verifyAfterParse = true);

llvm::Expected<SmallVector<Value>> parseSourceStringIntoBlock(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    TypeRange resultTypes,
    StringRef sourceString,
    Block *block);

}
