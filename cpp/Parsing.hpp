#pragma once

#include "Ops.hpp"
#include <llvm/Support/Error.h>
#include <optional>
#include <tuple>

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

llvm::Expected<SmallVector<Value>> parseSourceStringIntoBlock(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    ArrayRef<StringRef> typeAliasNames,
    TypeRange typeAliasTypes,
    TypeRange resultTypes,
    StringRef sourceString,
    Block *block);

}
