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

MlirOperation inlineInlineRegionOpParseFromSourceString(
    MlirLocation loc,
    MlirStringRef* operandNames,
    MlirValue* operands,
    intptr_t numInputs,
    MlirStringRef sourceString,
    MlirLocation* errorLoc,
    char* errorMessageBuffer,
    intptr_t errorMessageBufferCapacity);

MlirOperation inlineYieldOpCreate(MlirLocation loc, MlirValue* results, intptr_t numResults);

#ifdef __cplusplus
}
#endif
