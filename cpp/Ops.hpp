#pragma once

#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

#define GET_OP_CLASSES
#include "Ops.hpp.inc"

namespace mlir::inline_ {

struct InlineRegionParseError : public llvm::ErrorInfo<InlineRegionParseError> {
  static char ID;

  std::string message;
  Location loc;

  inline InlineRegionParseError(std::string message, Location loc)
    : message(message), loc(loc) {}

  inline std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  inline void log(llvm::raw_ostream &os) const override {
    os << message;
  }
};

llvm::Expected<InlineRegionOp> parseInlineRegionOpFromSourceString(
    Location loc,
    ArrayRef<StringRef> operandNames,
    ValueRange operands,
    StringRef sourceString);

}
