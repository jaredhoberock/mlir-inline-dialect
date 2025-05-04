// RUN: opt --convert-to-llvm %s | FileCheck %s

// ---- Test 1: no result

// CHECK-LABEL: llvm.func @no_result
// CHECK: llvm.add
// CHECK-NEXT: llvm.return
func.func @no_result(%a: i32) -> () {
  inline.inline_region %a : (i32) -> () {
    %_ = arith.addi %a, %a : i32
  }
  return
}

// ---- Test 1: single result

// CHECK-LABEL: llvm.func @single_result
// CHECK:         %[[ADD:.*]] = llvm.add
// CHECK-NEXT:    %[[TWO:.*]] = llvm.mlir.constant(2
// CHECK-NEXT:    %[[MUL:.*]] = llvm.mul %[[ADD]], %[[TWO]]
// CHECK-NEXT:    llvm.return %[[MUL]]
func.func @single_result(%a: i32, %b: i32) -> i32 {
  %result = inline.inline_region %a, %b : (i32, i32) -> i32 {
    %sum = arith.addi %a, %b : i32
    %two = arith.constant 2 : i32
    %scaled = arith.muli %sum, %two : i32
    yield %scaled : i32
  }
  return %result : i32
}

// ---- Test 1: multiple results

// CHECK-LABEL: llvm.func @multi_result
// CHECK: llvm.add
// CHECK: llvm.sub
// CHECK: llvm.return
func.func @multi_result(%a: i32, %b: i32) -> (i32, i32) {
  %0, %1 = inline.inline_region %a, %b : (i32, i32) -> (i32, i32) {
    %sum = arith.addi %a, %b : i32
    %diff = arith.subi %a, %b : i32
    yield %sum, %diff : i32, i32
  }
  return %0, %1 : i32, i32
}
