func.func @no_result(%a: i32) -> () {
  inline.inline_region %a : (i32) -> () {
    %_ = arith.addi %a, %a : i32
  }
  return
}
