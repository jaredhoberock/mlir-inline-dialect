#ifndef INLINE_DIALECT_TD
#define INLINE_DIALECT_TD

include "mlir/IR/OpBase.td"

def InlineDialect : Dialect {
  let name = "inline";
  let summary = "A dialect for embedding inline MLIR regions with explicitly bound arguments";
  let cppNamespace = "::mlir::inline_";
}

class InlineOp<string mnemonic, list<Trait> traits = []> :
  Op<InlineDialect, mnemonic, traits>;

#endif // INLINE_DIALECT_TD
