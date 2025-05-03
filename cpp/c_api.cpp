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

} // end extern "C"
