#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::inline_ {

struct InlineRegionOpLowering : public OpConversionPattern<InlineRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(InlineRegionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Get the region to inline
    Region& sourceRegion = op.getRegion();
    Block& sourceBlock = sourceRegion.front();
    
    // Check for terminator
    auto yieldOp = dyn_cast<YieldOp>(sourceBlock.getTerminator());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(op, "expected inline.yield terminator");
    
    // Clone all operations (except terminator) before our op
    rewriter.setInsertionPoint(op);
    
    // Map block arguments to input values
    IRMapping mapper;
    for (auto [blockArg, operand] : llvm::zip(sourceBlock.getArguments(), 
                                             adaptor.getInputs())) {
      mapper.map(blockArg, operand);
    }
    
    // Clone all operations except the terminator
    for (auto& nestedOp : sourceBlock.without_terminator()) {
      rewriter.clone(nestedOp, mapper);
    }
    
    // Replace the original op with the mapped yield values
    SmallVector<Value> results;
    for (auto result : yieldOp.getResults()) {
      results.push_back(mapper.lookupOrDefault(result));
    }
    
    rewriter.replaceOp(op, results);
    return success();
  }
};

void populateInlineToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                            RewritePatternSet& patterns) {
  patterns.add<InlineRegionOpLowering>(typeConverter, patterns.getContext());
}

}
