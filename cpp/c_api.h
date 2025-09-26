#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

void inlineRegisterDialect(MlirContext ctx);

MlirOperation inlineInlineRegionOpCreate(MlirLocation loc,
                                         MlirValue* inputs, intptr_t numInputs,
                                         MlirType* resultTypes, intptr_t numResults,
                                         MlirRegion region);

struct InlineValueList {
  // true if parsing and inlining succeeded
  bool succeeded;

  // number of values yielded. only meaningful if succeeded
  intptr_t count;

  // array of yielded MlirValues; null if !succeeded
  // the caller is responsible for freeing values using free()
  MlirValue *values;
};

InlineValueList inlineParseSourceStringIntoBlock(
    MlirLocation loc,
    MlirStringRef* operandNames,
    MlirValue* operands,
    intptr_t numOperands,
    MlirType* resultTypes,
    intptr_t numResultTypes,
    MlirStringRef sourceString,
    MlirBlock block,
    size_t* errorLine,
    size_t* errorCol,
    size_t* errorByteOffset,
    char* errorMessageBuffer,
    intptr_t errorMessageBufferCapacity);

MlirOperation inlineYieldOpCreate(MlirLocation loc, MlirValue* results, intptr_t numResults);

#ifdef __cplusplus
}
#endif
