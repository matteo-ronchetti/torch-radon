#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void transformKernel(float* output, cudaTextureObject_t texObj) {
    // Calculate texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    output[y * 128 + x] = tex2D<float>(texObj, x+0.5, y+0.5);
}

void copy_image_cuda(float* x, float* y){
    cudaArray* tmp;
    auto my_tex = create_texture(x, tmp, 128, 128, 128);

    // Invoke kernel
    dim3 dimGrid(8, 8);
    dim3 dimBlock(16, 16);

    transformKernel<<<dimGrid, dimBlock>>>(y, my_tex);
}

int main(){
    std::cout << "Hello CUDA" << std::endl;
}
