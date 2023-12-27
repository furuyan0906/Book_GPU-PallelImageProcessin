
TARGET := a.out

TOP_DIR := .

SRC_DIR := $(TOP_DIR)/src
INC_DIR := $(TOP_DIR)
LIB_DIR := $(TOP_DIR)/lib
BIN_DIR := $(TOP_DIR)/bin
OBJ_DIR := $(TOP_DIR)/obj
TEST_DIR := $(TOP_DIR)/tests
LOG_DIR := $(TOP_DIR)/tests/log

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:%=$(OBJ_DIR)/%.o)

INCLUDE := -I$(SRC_DIR) -I$(INC_DIR)

LIBS :=
#THIDR_LIBS := -lglfw -lGLESv2
THIDR_LIBS := -lglut -lGL
LIB_PATH := -L${HOME}/Library/lib

DEFINES :=

CXX := g++
CXXFLAGS := -g -O3 -std=c++2a -Wall -Wextra -MMD -MP
CPPFLAGS := $(DEFINES) $(INCLUDE)
LDFLAGS := $(LIB_PATH) $(LIBS) $(THIDR_LIBS)

MKDIR := mkdir -p
RM := rm -r
ECHO := echo
BEAR := bear --

LSP_JSON := ./compile_commands.json
LSP_CACHE := ./.cache

.PHONY: run clean build lsp

all: build

run: $(BIN_DIR)/$(TARGET)
	$(BIN_DIR)/$(TARGET)

clean:
	-@$(RM) $(BIN_DIR) $(OBJ_DIR)

build: $(BIN_DIR)/$(TARGET)

$(BIN_DIR)/$(TARGET): $(BIN_DIR) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(CPPFLAGS)

$(OBJ_DIR)/%.cpp.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDFLAGS) $(CPPFLAGS)

$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

lsp: clean
ifneq ("$(wildcard $(LSP_JSON))", "")
	@$(RM) $(LSP_JSON)
endif
ifneq ("$(wildcard $(LSP_CACHE))", "")
	@$(RM) $(LSP_CACHE)
endif
ifneq ("$(shell which bear)", "")
	-@$(BEAR) "make"
else
	@$(ECHO) "Please install bear"
endif

