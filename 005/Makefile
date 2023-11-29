
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

INCLUDE := -I$(SRC_DIR) -I$(INC_DIR) -I${HOME}/Library/include

LIBS :=
THIDR_LIBS := -lglut -lGL
LIB_PATH := -L${HOME}/Library/lib

DEFINES :=

CXX := g++
CXXFLAGS := -g -O3 -std=c++2a -Wall -Wextra -MMD -MP
CPPFLAGS := $(DEFINES) $(INCLUDE)
LDFLAGS := $(LIB_PATH) $(LIBS) $(THIDR_LIBS)

MKDIR := mkdir -p
RM := rm -r


.PHONY: run clean

all: run

run: $(BIN_DIR)/$(TARGET)
	$(BIN_DIR)/$(TARGET)

$(BIN_DIR)/$(TARGET): $(BIN_DIR) $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(CPPFLAGS)

clean:
	-@$(RM) $(BIN_DIR)/$(TARGET) $(OBJ_DIR)

$(OBJ_DIR)/%.cpp.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDFLAGS) $(CPPFLAGS)

$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

