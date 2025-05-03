#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace inline_ {

void populateInlineToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                            RewritePatternSet& patterns);
}
}
