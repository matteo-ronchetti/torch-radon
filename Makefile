NVCC := /usr/local/cuda/bin/nvcc
CXX := g++

LIBRARIES := curand

INCLUDE_DIRS := ./include/

FLAGS := -std=c++11
FLAGS += -ccbin=$(CXX) -Xcompiler -fPIC
FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

# compile for K80, 1070, V100, T4
# FLAGS += -gencode arch=compute_35,code=sm_35
# FLAGS += -gencode arch=compute_61,code=sm_61
FLAGS += -gencode arch=compute_70,code=sm_70
FLAGS += -gencode arch=compute_75,code=sm_75
FLAGS += -DNDEBUG -O3 --generate-line-info

SRC_DIR := ./src
OBJ_DIR := ./objs
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))

all: $(OBJ_DIR)/cuda/libradon.a | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $@
	mkdir -p $@/cuda

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c $< -o $@

$(OBJ_DIR)/cuda/libradon.a: $(CU_OBJS)
	ar rc $@ $(ALL_OBJS) $(CU_OBJS)

install: all
	rm -r build || true
	rm -r dist || true
	python setup.py install

clean:
	rm -r $(OBJ_DIR)
