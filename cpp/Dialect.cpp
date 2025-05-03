#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include "Dialect.cpp.inc"

namespace mlir::inline_ {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateInlineToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void InlineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  addInterfaces<
    ConvertToLLVMInterface
  >();
}

}
