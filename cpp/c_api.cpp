#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
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

MlirOperation inlineYieldOpCreate(MlirLocation loc, MlirValue *results, intptr_t numResults) {
  MLIRContext *context = unwrap(loc)->getContext();
  OpBuilder builder(context);
  OperationState state(unwrap(loc), YieldOp::getOperationName());

  for (intptr_t i = 0; i < numResults; ++i)
    state.operands.push_back(unwrap(results[i]));

  return wrap(builder.create(state));
}

} // end extern "C"
