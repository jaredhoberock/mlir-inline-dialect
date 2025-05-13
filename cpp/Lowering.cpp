#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::inline_ {

struct InlineRegionOpLowering : public OpConversionPattern<InlineRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(InlineRegionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // inline the op's region and get the results
    SmallVector<Value,4> results;
    std::string errorMessage;
    if (failed(op.cloneBodyAtInsertionPoint(rewriter, adaptor.getInputs(), results, errorMessage))) {
      return rewriter.notifyMatchFailure(op, errorMessage);
    }
    
    // replace the original op with its results
    rewriter.replaceOp(op, results);
    return success();
  }
};

void populateInlineToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                            RewritePatternSet& patterns) {
  patterns.add<InlineRegionOpLowering>(typeConverter, patterns.getContext());
}

}
