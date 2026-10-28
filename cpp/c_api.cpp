#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include "Parsing.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Builders.h>

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
    InlineParseError* error) {
  SmallVector<StringRef> names, aliasNames;
  SmallVector<Value> values;
  SmallVector<Type> aliasTypes, types;
  llvm::Expected<SmallVector<Value>> result = parseSourceStringIntoBlock(
      unwrap(loc),
      unwrapList(numOperands, operandNames, names),
      unwrapList(numOperands, operands, values),
      unwrapList(numTypeAliases, typeAliasNames, aliasNames),
      unwrapList(numTypeAliases, typeAliasTypes, aliasTypes),
      unwrapList(numResultTypes, resultTypes, types),
      unwrap(sourceString), unwrap(block), verifyAfterParse);

  if (!result) {
    llvm::handleAllErrors(result.takeError(), [&](const InlineRegionParseError &e) {
      TextPosition position = e.position.value_or(TextPosition{0, 0, 0});
      *error = {strdup(e.message.c_str()), e.position.has_value(),
                position.line, position.column, position.byteOffset};
    });
    return {false, 0, nullptr};
  }

  intptr_t count = result->size();
  auto *wrapped = count ? static_cast<MlirValue*>(malloc(sizeof(MlirValue) * count)) : nullptr;
  for (auto [i, value] : llvm::enumerate(*result))
    wrapped[i] = wrap(value);
  return {true, count, wrapped};
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
