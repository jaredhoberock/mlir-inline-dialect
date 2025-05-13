#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <llvm/Support/SourceMgr.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
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


void extractInlineParseErrorLocationAndMessage(
    llvm::Error &&err,
    size_t *errorLine,
    size_t *errorCol,
    size_t *errorByteOffset,
    char *errorMessageBuffer,
    intptr_t errorMessageBufferCapacity) {

  std::string errorMessage;
  size_t line = 0, col = 0;
  size_t byteOffset = static_cast<size_t>(-1);  // sentinel for unknown offset

  llvm::handleAllErrors(std::move(err), [&](const InlineRegionParseError &e) {
    errorMessage = e.message;
    if (auto loc = e.getLocation()) {
      std::tie(line, col, byteOffset) = *loc;
    }
  });

  if (errorLine) *errorLine = line;
  if (errorCol) *errorCol = col;
  if (errorByteOffset) *errorByteOffset = byteOffset;

  if (errorMessageBuffer && errorMessageBufferCapacity > 0) {
    std::snprintf(errorMessageBuffer, errorMessageBufferCapacity,
                  "%s", errorMessage.c_str());
  }
}


MlirOperation inlineInlineRegionOpParseFromSourceString(
    MlirLocation wrappedLoc,
    MlirStringRef* wrappedOperandNames,
    MlirValue* wrappedOperands,
    intptr_t numOperands,
    MlirStringRef wrappedSourceString,
    size_t *errorLine,
    size_t *errorCol,
    size_t *errorByteOffset,
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
    sourceString,
    /*verifyAfterParse=*/false
  );

  if (result)
    return wrap(*result);

  // get the error information
  extractInlineParseErrorLocationAndMessage(result.takeError(),
                                            errorLine, errorCol, errorByteOffset,
                                            errorMessageBuffer, errorMessageBufferCapacity);

  return MlirOperation{nullptr};
}


InlineValueList inlineParseSourceStringIntoBlock(
    MlirLocation wrappedLoc,
    MlirStringRef* wrappedOperandNames,
    MlirValue* wrappedOperands,
    intptr_t numOperands,
    MlirType* wrappedResultTypes,
    intptr_t numResultTypes,
    MlirStringRef wrappedSourceString,
    MlirBlock wrappedBlock,
    size_t* errorLine,
    size_t* errorCol,
    size_t* errorByteOffset,
    char* errorMessageBuffer,
    intptr_t errorMessageBufferCapacity) {

  InlineValueList out = {
    .succeeded = false,
    .count = 0,
    .values = nullptr
  };

  Location loc = unwrap(wrappedLoc);
  Block* block = unwrap(wrappedBlock);

  // unwrap inputs
  SmallVector<StringRef,4> operandNames;
  SmallVector<Value,4> operandValues;
  for (intptr_t i = 0; i < numOperands; ++i) {
    operandNames.push_back(StringRef(wrappedOperandNames[i].data, wrappedOperandNames[i].length));
    operandValues.push_back(unwrap(wrappedOperands[i]));
  }

  // wrap result types
  SmallVector<Type,4> resultTypes;
  for (intptr_t i = 0; i < numResultTypes; ++i) {
    resultTypes.push_back(unwrap(wrappedResultTypes[i]));
  }

  StringRef sourceString = StringRef(wrappedSourceString.data, wrappedSourceString.length);

  // call C++
  llvm::Expected<SmallVector<Value>> result =
    parseSourceStringIntoBlock(loc, operandNames, operandValues, resultTypes, sourceString, block);

  // handle error case
  if (!result) {
    extractInlineParseErrorLocationAndMessage(result.takeError(),
                                              errorLine, errorCol, errorByteOffset,
                                              errorMessageBuffer, errorMessageBufferCapacity);
    return out;
  }

  // success: wrap results into MlirValue[]
  const SmallVector<Value> &values = *result;
  out.succeeded = true;
  out.count = static_cast<intptr_t>(values.size());
  out.values = out.count > 0
    ? static_cast<MlirValue*>(malloc(sizeof(MlirValue) * out.count))
    : nullptr;

  for (intptr_t i = 0; i < out.count; ++i)
    out.values[i] = wrap(values[i]);

  return out;
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
