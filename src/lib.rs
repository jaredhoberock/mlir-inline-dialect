use melior::{ir::{Location, Operation, Region, Type, TypeLike, Value, ValueLike}, Context};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirRegion, MlirType, MlirValue};

#[link(name = "inline_dialect")]
unsafe extern "C" {
    fn inlineRegisterDialect(ctx: MlirContext);
    fn inlineInlineRegionOpCreate(loc: MlirLocation,
                                  inputs: *const MlirValue, num_inputs: isize,
                                  result_types: *const MlirType, num_results: isize,
                                  region: MlirRegion) -> MlirOperation;
    fn inlineYieldOpCreate(loc: MlirLocation,
                           results: *const MlirValue, num_results: isize) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { inlineRegisterDialect(context.to_raw()) }
}

pub fn inline_region<'c>(
    loc: Location<'c>,
    inputs: &[Value<'c,'_>],
    result_types: &[Type<'c>],
    region: Region<'c>
) -> Operation<'c> {
    unsafe {
        let raw_inputs: Vec<MlirValue> = inputs.iter().map(|v| v.to_raw()).collect();
        let raw_result_types: Vec<MlirType> = result_types.iter().map(|t| t.to_raw()).collect();

        let raw_op = inlineInlineRegionOpCreate(
            loc.to_raw(),
            raw_inputs.as_ptr(),
            raw_inputs.len() as isize,
            raw_result_types.as_ptr(),
            raw_result_types.len() as isize,
            region.into_raw(),
        );

        Operation::from_raw(raw_op)
    }
}

pub fn yield_<'c>(loc: Location<'c>, results: &[Value<'c,'_>]) -> Operation<'c> {
    unsafe {
        let raw_results: Vec<MlirValue> = results.iter().map(|v| v.to_raw()).collect();
        let raw_op = inlineYieldOpCreate(
            loc.to_raw(), 
            raw_results.as_ptr(), 
            raw_results.len() as isize,
        );
        Operation::from_raw(raw_op)
    }
}
