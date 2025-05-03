use melior::{
    Context,
    dialect::{arith, func, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        Attribute,
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, Region, RegionLike,
    },
    pass::{self, PassManager},
    utility::{register_all_dialects},
};
use inline_dialect as inline;


#[test]
fn test_inline_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    inline::register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // begin creating a module
    let loc = Location::unknown(&context);
    let mut module = Module::new(loc);

    // build a func.func @test:
    //
    //  func.func @test(%arg0: i32, %arg1: i32) -> i32 {
    //    %0 = inline.inline_region %arg0, %arg1 : (i32, i32) -> i32 {
    //      %1 = arith.addi %arg0, %arg1 : i32
    //      inline.yield %1 : i32
    //    }
    //    return %0 : i32
    //  }
    let i32_ty = IntegerType::new(&context, 32).into();

    // Build the function body
    let region = {
        let inner_region = {
            let inner_block = Block::new(&[(i32_ty, loc), (i32_ty, loc)]);
            let inner_result = inner_block.append_operation(arith::addi(
                inner_block.argument(0).unwrap().into(),
                inner_block.argument(1).unwrap().into(),
                loc,
            ));
            inner_block.append_operation(inline::yield_(
                loc,
                &[inner_result.result(0).unwrap().into()],
            ));

            let inner_region = Region::new();
            inner_region.append_block(inner_block);
            inner_region
        };

        let block = Block::new(&[(i32_ty, loc), (i32_ty, loc)]);
        let result = block.append_operation(inline::inline_region(
            loc,
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i32_ty],
            inner_region,
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            loc,
        ));

        let region = Region::new();
        region.append_block(block);
        region
    };

    // Define the function
    let function_type = FunctionType::new(
        &context, 
        &[i32_ty, i32_ty],
        &[i32_ty]
    );
    let mut func_op = func::func(
        &context,
        StringAttribute::new(&context, "test"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        loc,
    );

    // this attribute tells MLIR to create an additional wrapper function that we can use 
    // to invoke "test" via invoke_packed below
    func_op.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));
    module.body().append_operation(func_op);

    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test that we can call the function and it produces the expected result
    unsafe {
        let mut a: i32 = 7;
        let mut b: i32 = 13;
        let mut result: i32 = 0;

        let mut packed_args: [*mut (); 3] = [
            &mut a as *mut i32 as *mut (),
            &mut b as *mut i32 as *mut (),
            &mut result as *mut i32 as *mut (),
        ];

        engine.invoke_packed("test", &mut packed_args)
            .expect("JIT invocation failed");

        assert_eq!(result, 7 + 13);
    }
}
