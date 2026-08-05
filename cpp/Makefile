ifndef MLIR_SYS_220_PREFIX
$(error MLIR_SYS_220_PREFIX must be set; point it at your LLVM/MLIR 22 install)
endif

BUILD_DIR := build
LLVM_BIN_PATH := $(MLIR_SYS_220_PREFIX)/bin
LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen

CXX := clang++
# The install's llvm-config reports no optimization level; set -O2 as the
# default here. Asserts stay on (no -DNDEBUG) -- the dialect verifiers are product.
CXXFLAGS := -Wall -Werror -fPIC $(shell $(LLVM_CONFIG) --cxxflags) -O2
INCLUDES := -I$(BUILD_DIR) -I. -I$(MLIR_SYS_220_PREFIX)/include

DIALECT_SOURCES := c_api.cpp Dialect.cpp Lowering.cpp Ops.cpp Parsing.cpp
DIALECT_OBJECTS := $(addprefix $(BUILD_DIR)/,$(DIALECT_SOURCES:.cpp=.o))
PLUGIN_OBJECTS := $(DIALECT_OBJECTS) $(BUILD_DIR)/Plugin.o
DEPS := $(PLUGIN_OBJECTS:.o=.d)
TD_FILES := $(wildcard *.td)

GENERATED := \
  $(BUILD_DIR)/Dialect.hpp.inc \
  $(BUILD_DIR)/Dialect.cpp.inc \
  $(BUILD_DIR)/Ops.hpp.inc \
  $(BUILD_DIR)/Ops.cpp.inc

.PHONY: all clean test

all: $(BUILD_DIR)/libinline_dialect.a $(BUILD_DIR)/libinline_dialect.so

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/Dialect.hpp.inc: Dialect.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/Dialect.cpp.inc: Dialect.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/Ops.hpp.inc: Ops.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/Ops.cpp.inc: Ops.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(GENERATED) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(BUILD_DIR)/libinline_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

$(BUILD_DIR)/libinline_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

test: $(BUILD_DIR)/libinline_dialect.so
	INLINE_DIALECT_PLUGIN=$(CURDIR)/$(BUILD_DIR)/libinline_dialect.so lit tests

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
