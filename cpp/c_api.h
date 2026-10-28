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

// Parses `sourceString`, MLIR text in which each operand name and type alias
// name is bound, and appends its ops to `block`, yielding the text's yielded
// values. `loc` is the position of the text's first character in the file the
// text came from: each parsed op and block argument carries its position in
// that file, the text's first line continuing `loc`'s line and each later line
// starting at column 1 of a following line. A text whose ops were written with
// explicit locations keeps them. On failure, `errorLine` and `errorCol` give
// the error's position relative to the text (both from 1) and
// `errorByteOffset` its byte offset into the text, when the parser reports one.
InlineValueList inlineParseSourceStringIntoBlock(
    MlirLocation loc,
    MlirStringRef* operandNames,
    MlirValue* operands,
    intptr_t numOperands,
    MlirStringRef* typeAliasNames,
    MlirType* typeAliasTypes,
    intptr_t numTypeAliases,
    MlirType* resultTypes,
    intptr_t numResultTypes,
    MlirStringRef sourceString,
    MlirBlock block,
    bool verifyAfterParse,
    size_t* errorLine,
    size_t* errorCol,
    size_t* errorByteOffset,
    char* errorMessageBuffer,
    intptr_t errorMessageBufferCapacity);

MlirOperation inlineYieldOpCreate(MlirLocation loc, MlirValue* results, intptr_t numResults);

#ifdef __cplusplus
}
#endif
