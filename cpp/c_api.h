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

// A failure of inlineParseSourceStringIntoBlock.
struct InlineParseError {
  // the failure's message; the caller frees it with free()
  char *message;

  // whether line, column and byteOffset give the failure's position in the
  // text: line and column from 1, the column in bytes, and the byte offset
  // from the text's start. Only a parse error inside the text has one.
  bool hasPosition;
  size_t line;
  size_t column;
  size_t byteOffset;
};

// Parses `sourceString`, MLIR text in which each operand name is bound to its
// operand and each type alias name to its type, and appends its ops to
// `block`. A trailing `inline.yield` is not appended: its operands are the
// yielded values, one for each result type. `loc` is the position of the text's
// first character in the file the text came from: each appended op and block
// argument carries its position in that file, the text's first line continuing
// `loc`'s line and each later line starting at column 1 of a following line. A
// text whose ops were written with explicit locations keeps them. On failure,
// fills `*error`.
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
    struct InlineParseError* error);

MlirOperation inlineYieldOpCreate(MlirLocation loc, MlirValue* results, intptr_t numResults);

#ifdef __cplusplus
}
#endif
