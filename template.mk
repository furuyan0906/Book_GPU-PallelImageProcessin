
TARGET := a.out

TOP_DIR := .

SRC_DIR := $(TOP_DIR)/src
CUDA_DIR := $(TOP_DIR)/cuda
UTIL_DIR := $(TOP_DIR)/util
INC_DIR := $(TOP_DIR)
LIB_DIR := $(TOP_DIR)/lib
BIN_DIR := $(TOP_DIR)/bin
OBJ_DIR := $(TOP_DIR)/obj
TEST_DIR := $(TOP_DIR)/tests
LOG_DIR := $(TOP_DIR)/tests/log

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRCS := $(wildcard $(CUDA_DIR)/*.cu)

OBJS := $(SRCS:%=$(OBJ_DIR)/%.o)
CUDA_OBJS := $(CUDA_SRCS:%=$(OBJ_DIR)/%.o)

INCLUDE := -I$(SRC_DIR) -I$(CUDA_DIR) -I$(UTIL_DIR) -I$(INC_DIR) -I${HOME}/Library/include

LIBS :=
THIDR_LIBS := -lglut -lGL
LIB_PATH := -L${HOME}/Library/lib

DEFINES :=

CXXFLAGS := -g -mavx2 -O3 -std=c++2a -Wall -Wextra -MMD -MP
CPPFLAGS := $(DEFINES) $(INCLUDE)
LDFLAGS := $(LIB_PATH) $(LIBS) $(THIDR_LIBS)

NVCCFLAGS := -O3
CUDA_CUFLAGS := $(DEFINES) $(INCLUDE)
CUDA_LDFLAGS := -L${CUDA_INSTALL_PATH}/lib64 -lcudart

CXX := g++
NVCC := nvcc
MKDIR := mkdir -p
RM := rm -r


.PHONY: run clean

all: run

run: $(BIN_DIR)/$(TARGET)
	$(BIN_DIR)/$(TARGET)

$(BIN_DIR)/$(TARGET): $(BIN_DIR) $(OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(CUDA_OBJS) $(LDFLAGS) $(CUDA_LDFLAGS) $(CPPFLAGS)

clean:
	-@$(RM) $(BIN_DIR) $(OBJ_DIR)

$(OBJ_DIR)/%.cpp.o: %.cpp
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDFLAGS) $(CPPFLAGS)

$(OBJ_DIR)/%.cu.o: %.cu
	@$(MKDIR) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< $(CUDA_LDFLAGS) $(CUDA_CUFLAGS)

$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

