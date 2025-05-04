#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <llvm/Support/SourceMgr.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;
using namespace mlir::inline_;

extern "C" {

void inlineRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<InlineDialect>();
}

MlirOperation inlineInlineRegionOpCreate(MlirLocation loc,
                                         MlirValue* inputs, intptr_t numInputs,
                                         MlirType* resultTypes, intptr_t numResults,
                                         MlirRegion region) {
  MLIRContext *context = unwrap(loc)->getContext();
  OpBuilder builder(context);
  OperationState state(unwrap(loc), InlineRegionOp::getOperationName());

  for (intptr_t i = 0; i < numInputs; ++i)
    state.operands.push_back(unwrap(inputs[i]));

  for (intptr_t i = 0; i < numResults; ++i)
    state.types.push_back(unwrap(resultTypes[i]));

  state.addRegion(std::unique_ptr<Region>(unwrap(region)));

  return wrap(builder.create(state));
}

MlirOperation inlineInlineRegionOpParseFromSourceString(
    MlirLocation wrappedLoc,
    MlirStringRef* wrappedOperandNames,
    MlirValue* wrappedOperands,
    intptr_t numOperands,
    MlirStringRef wrappedSourceString,
    char *errorMessageBuffer,
    intptr_t errorMessageBufferCapacity) {

  Location loc = unwrap(wrappedLoc);
  MLIRContext* context = loc->getContext();

  // collect operand names and values
  SmallVector<StringRef> operandNames;
  SmallVector<Value> operands;
  for (intptr_t i = 0; i < numOperands; ++i) {
    operandNames.push_back(StringRef(wrappedOperandNames[i].data, wrappedOperandNames[i].length));
    operands.push_back(unwrap(wrappedOperands[i]));
  }

  // unwrap the source string
  StringRef sourceString = StringRef(wrappedSourceString.data, wrappedSourceString.length);

  // parse
  llvm::Expected<InlineRegionOp> result = parseInlineRegionOpFromSourceString(
    loc,
    operandNames,
    operands,
    sourceString
  );

  if (result)
    return wrap(*result);

  // copy any error message into the caller's buffer
  std::string errorStr;
  llvm::handleAllErrors(result.takeError(), [&](llvm::ErrorInfoBase &eib) {
    errorStr = eib.message();
  });

  if (errorMessageBuffer && errorMessageBufferCapacity > 0) {
    std::snprintf(errorMessageBuffer, errorMessageBufferCapacity, "%s", errorStr.c_str());
  }

  return MlirOperation{nullptr};
}

MlirOperation inlineYieldOpCreate(MlirLocation loc, MlirValue *results, intptr_t numResults) {
  MLIRContext *context = unwrap(loc)->getContext();
  OpBuilder builder(context);
  OperationState state(unwrap(loc), YieldOp::getOperationName());

  for (intptr_t i = 0; i < numResults; ++i)
    state.operands.push_back(unwrap(results[i]));

  return wrap(builder.create(state));
}

} // end extern "C"
