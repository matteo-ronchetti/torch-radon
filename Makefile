NVCC := /usr/local/cuda/bin/nvcc
CXX := g++

LIBRARIES := curand

INCLUDE_DIRS := ./include/

CXX_FLAGS := -std=c++11 -fPIC $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) -O3

FLAGS := -std=c++11
FLAGS += -ccbin=$(CXX) -Xcompiler -fPIC
FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

# compile for K80, 1070, V100, T4
FLAGS += -gencode arch=compute_35,code=sm_35
FLAGS += -gencode arch=compute_61,code=sm_61
FLAGS += -gencode arch=compute_70,code=sm_70
FLAGS += -gencode arch=compute_75,code=sm_75
FLAGS += -DNDEBUG -O3 --generate-line-info --compiler-options -Wall

# FLAGS += -DVERBOSE
# CXX_FLAGS += -DVERBOSE

SRC_DIR := ./src
OBJ_DIR := ./objs
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS := ./src/cache.cpp #$(wildcard $(SRC_DIR)/*.cpp)
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))


all: $(OBJ_DIR)/cuda/libradon.a | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $@
	mkdir -p $@/cuda

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/cuda/libradon.a: $(CU_OBJS) $(CPP_OBJS)
	ar rc $@ $(ALL_OBJS) $(CU_OBJS) $(CPP_OBJS)

install: all
	rm -r build || true
	rm -r dist || true
	python setup.py install

clean:
	rm -r $(OBJ_DIR)
