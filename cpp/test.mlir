func.func @no_result(%a: i32) -> () {
  inline.inline_region %a : (i32) -> () {
    %_ = arith.addi %a, %a : i32
  }
  return
}

func.func @single_result(%a: i32, %b: i32) -> i32 {
  %result = inline.inline_region %a, %b : (i32, i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    %two = arith.constant 2 : i32
    %scaled = arith.muli %sum, %two : i32
    inline.yield %scaled : i32
  }
  return %result : i32
}

func.func @multi_result(%a: i32, %b: i32) -> (i32, i32) {
  %0, %1 = inline.inline_region %a, %b : (i32, i32) -> (i32, i32) {
    %sum = arith.addi %a, %b : i32
    %diff = arith.subi %a, %b : i32
    inline.yield %sum, %diff : i32, i32
  }
  return %0, %1 : i32, i32
}
