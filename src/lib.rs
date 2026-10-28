use melior::{ir::{Block, Location, Operation, Region, Type, TypeLike, Value, ValueLike}, Context, StringRef};
use mlir_sys::{MlirBlock, MlirContext, MlirLocation, MlirOperation, MlirRegion, MlirStringRef, MlirType, MlirValue};
use thiserror::Error;

#[repr(C)]
#[derive(Debug)]
struct InlineValueList {
    succeeded: bool,
    count: isize,
    values: *mut MlirValue,
}

#[repr(C)]
struct InlineParseError {
    message: *mut std::ffi::c_char,
    has_position: bool,
    line: usize,
    column: usize,
    byte_offset: usize,
}

#[link(name = "inline_dialect")]
unsafe extern "C" {
    fn inlineRegisterDialect(ctx: MlirContext);
    fn inlineInlineRegionOpCreate(loc: MlirLocation,
                                  inputs: *const MlirValue, num_inputs: isize,
                                  result_types: *const MlirType, num_results: isize,
                                  region: MlirRegion) -> MlirOperation;
    fn inlineYieldOpCreate(loc: MlirLocation,
                           results: *const MlirValue, num_results: isize) -> MlirOperation;

    fn inlineParseSourceStringIntoBlock(
        loc: MlirLocation,
        operand_names: *const MlirStringRef,
        operand_value: *const MlirValue,
        num_operands: isize,
        type_alias_names: *const MlirStringRef,
        type_alias_types: *const MlirType,
        num_type_aliases: isize,
        result_types: *const MlirType,
        num_result_types: isize,
        source_string: MlirStringRef,
        block: MlirBlock,
        verify_after_parse: bool,
        error: *mut InlineParseError) -> InlineValueList;
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

#[derive(Debug, Error)]
#[error("{message}")]
pub struct Error {
    pub message: String,
    pub line_col_byte_offset: Option<(usize, usize, usize)>,
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

pub fn parse_source_into_block<'c,'b>(
    loc: Location<'c>,
    operands: &[(StringRef, Value<'c,'_>)],
    type_aliases: &[(StringRef, Type<'c>)],
    result_types: &[Type<'c>],
    src_str: StringRef,
    block: &'b Block<'c>,
    verify_after_parse: bool,
) -> Result<Vec<Value<'c,'b>>, Error> {
    unsafe {
        let (operand_names, operand_values): (Vec<_>, Vec<_>) = operands.iter()
            .map(|(n, v)| (n.to_raw(), v.to_raw()))
            .unzip();

        let (alias_names, alias_types): (Vec<_>, Vec<_>) = type_aliases.iter()
            .map(|(n, t)| (n.to_raw(), t.to_raw()))
            .unzip();
        
        let results: Vec<_> = result_types.iter()
            .map(|ty| ty.to_raw())
            .collect();

        let mut error = InlineParseError {
            message: std::ptr::null_mut(),
            has_position: false,
            line: 0,
            column: 0,
            byte_offset: 0,
        };

        let result = inlineParseSourceStringIntoBlock(
            loc.to_raw(),
            operand_names.as_ptr(),
            operand_values.as_ptr(),
            operands.len() as isize,
            alias_names.as_ptr(),
            alias_types.as_ptr(),
            type_aliases.len() as isize,
            results.as_ptr(),
            results.len() as isize,
            src_str.to_raw(),
            block.to_raw(),
            verify_after_parse,
            &mut error,
        );

        if !result.succeeded {
            let message = std::ffi::CStr::from_ptr(error.message)
                .to_string_lossy()
                .into_owned();
            libc::free(error.message.cast());
            let line_col_byte_offset = error.has_position
                .then_some((error.line, error.column, error.byte_offset));
            return Err(Error { message, line_col_byte_offset });
        }

        let values = if result.count == 0 {
            Vec::new()
        } else {
            let slice = std::slice::from_raw_parts(result.values, result.count as usize);
            let vec = slice.iter().map(|&v| Value::from_raw(v)).collect();
            libc::free(result.values.cast());
            vec
        };

        Ok(values)
    }
}
