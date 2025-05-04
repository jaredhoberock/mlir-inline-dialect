use melior::{ir::{Location, Operation, Region, Type, TypeLike, Value, ValueLike}, Context, StringRef};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirRegion, MlirStringRef, MlirType, MlirValue};

#[link(name = "inline_dialect")]
unsafe extern "C" {
    fn inlineRegisterDialect(ctx: MlirContext);
    fn inlineInlineRegionOpCreate(loc: MlirLocation,
                                  inputs: *const MlirValue, num_inputs: isize,
                                  result_types: *const MlirType, num_results: isize,
                                  region: MlirRegion) -> MlirOperation;
    fn inlineInlineRegionOpParseFromSourceString(
        loc: MlirLocation,
        operand_names: *const MlirStringRef,
        operand_values: *const MlirValue,
        num_operands: isize,
        source_string: MlirStringRef,
        error_message: *mut std::ffi::c_char,
        error_message_buffer_capcity: isize) -> MlirOperation;
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

pub fn parse_inline_region<'c>(
    loc: Location<'c>,
    operands: &[(StringRef, Value<'c,'_>)],
    src_str: StringRef,
) -> Result<Operation<'c>,String> {
    unsafe {
        let raw_operand_names: Vec<MlirStringRef> = operands.iter().map(|(n, _)| n.to_raw()).collect();
        let raw_operand_values: Vec<MlirValue> = operands.iter().map(|(_, v)| v.to_raw()).collect();

        // allocate a fixed-size error message buffer
        const ERROR_MESSAGE_BUFFER_CAPACITY: usize = 1024;
        let mut error_buf = [0 as std::ffi::c_char; ERROR_MESSAGE_BUFFER_CAPACITY];

        let raw_op = inlineInlineRegionOpParseFromSourceString(
            loc.to_raw(),
            raw_operand_names.as_ptr(),
            raw_operand_values.as_ptr(),
            operands.len() as isize,
            src_str.to_raw(),
            error_buf.as_mut_ptr(),
            ERROR_MESSAGE_BUFFER_CAPACITY as isize,
        );

        if raw_op.ptr.is_null() {
            // convert the c string to Rust String
            let cstr = std::ffi::CStr::from_ptr(error_buf.as_ptr());
            return Err(cstr.to_string_lossy().into_owned());
        }

        Ok(Operation::from_raw(raw_op))
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
