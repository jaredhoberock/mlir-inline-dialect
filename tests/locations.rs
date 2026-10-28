use std::cell::RefCell;

use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Block, BlockLike, Location, Module, Region, RegionLike, r#type::IntegerType},
    ir::operation::OperationLike,
    utility::register_all_dialects,
    StringRef,
};
use inline_dialect as inline;

/// Ops parsed from inlined text carry the positions they occupy in the file
/// the text came from: the caller's location names the text's first
/// character, the first text line continues that line, and later text lines
/// start at column 1 of the following file lines.
#[test]
fn parsed_ops_carry_file_positions() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    inline::register(&context);
    context.load_all_available_dialects();

    let i32_ty = IntegerType::new(&context, 32).into();
    let unknown = Location::unknown(&context);
    let module = Module::new(unknown);
    let region = Region::new();
    let block = region.append_block(Block::new(&[(i32_ty, unknown)]));
    let operand = block.argument(0).unwrap().into();

    // The text's first character lies at line 10, column 7 of prog.cuw, and an
    // op's position is that of its name. The second text line holds an op that
    // parses but fails verification, and the third a nested region whose op
    // lies on the fourth.
    let text = "%c = arith.addi %a, %a : i32\n  %d = \"arith.addi\"(%c, %c) : (i32, i32) -> i64\n%e = scf.execute_region -> i32 {\n    scf.yield %c : i32\n}\n";
    inline::parse_source_into_block(
        Location::new(&context, "prog.cuw", 10, 7),
        &[(StringRef::new("a"), operand)],
        &[],
        &[],
        StringRef::new(text),
        &block,
        false,
    )
    .expect("the text parses");

    let first = block.first_operation().unwrap();
    let second = first.next_in_block().unwrap();
    let third = second.next_in_block().unwrap();
    let nested = third.region(0).unwrap().first_block().unwrap().first_operation().unwrap();
    assert_eq!(first.location().to_string(), "loc(\"prog.cuw\":10:12)");
    assert_eq!(second.location().to_string(), "loc(\"prog.cuw\":11:8)");
    assert_eq!(third.location().to_string(), "loc(\"prog.cuw\":12:6)");
    assert_eq!(nested.location().to_string(), "loc(\"prog.cuw\":13:5)");

    // Verification reports the failing op at its position in prog.cuw.
    let locations = RefCell::new(Vec::new());
    let handler = context.attach_diagnostic_handler(|diagnostic| {
        locations.borrow_mut().push(diagnostic.location().to_string());
        true
    });
    let func = melior::dialect::func::func(
        &context,
        melior::ir::attribute::StringAttribute::new(&context, "f"),
        melior::ir::attribute::TypeAttribute::new(
            melior::ir::r#type::FunctionType::new(&context, &[i32_ty], &[]).into(),
        ),
        region,
        &[],
        unknown,
    );
    let func = module.body().append_operation(func);
    assert!(!func.verify());
    context.detach_diagnostic_handler(handler);
    assert_eq!(locations.into_inner(), vec!["loc(\"prog.cuw\":11:8)".to_string()]);
}

/// A parse error reports its position relative to the text: line and column
/// from 1 at the text's first character, and the byte offset into the text.
#[test]
fn parse_errors_carry_text_positions() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    inline::register(&context);
    context.load_all_available_dialects();

    let i32_ty = IntegerType::new(&context, 32).into();
    let unknown = Location::unknown(&context);
    let block = Block::new(&[(i32_ty, unknown)]);
    let operand = block.argument(0).unwrap().into();

    let text = "%c = arith.addi %a, %a : i32\n  %d = arith.addi %c, %a : i64\n";
    let error = inline::parse_source_into_block(
        Location::new(&context, "prog.cuw", 10, 7),
        &[(StringRef::new("a"), operand)],
        &[],
        &[],
        StringRef::new(text),
        &block,
        false,
    )
    .expect_err("the text uses %c at a type it does not have");
    assert_eq!(error.line_col_byte_offset, Some((2, 19, 47)));
}
