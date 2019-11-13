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
# FLAGS += -gencode arch=compute_70,code=sm_70
FLAGS += -gencode arch=compute_75,code=sm_75

SRC_DIR := ./src
OBJ_DIR := ./objs
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))

all: $(OBJ_DIR)/cuda/libradon.so | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $@
	mkdir -p $@/cuda

$(OBJ_DIR)/cuda/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -dc $< -o $@

$(OBJ_DIR)/cuda/libradon.so: $(CU_OBJS)
	#gcc -fPIC -shared -o $@ -L/usr/local/cuda/lib64 -lcuda -lcudart $(CU_OBJS)
	$(NVCC) $(FLAGS) -dlink $(CU_OBJS) -o radon.o
	g++ -shared $(CU_OBJS) radon.o -o $@


install: all
	rm -r build || true
	rm -r dist || true
	python setup.py install

clean:
	rm -r $(OBJ_DIR)
