#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void radon_forward_kernel(float* output, cudaTextureObject_t texObj, const float* rays, const float* angles, const int n_rays, const int n_angles) {
    // Calculate texture coordinates
    unsigned int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const float rsx = rays[ray_id*4+0];
    const float rsy = rays[ray_id*4+1];
    const float rex = rays[ray_id*4+2];
    const float rey = rays[ray_id*4+3];
    
    for(int i = 0; i < n_angles; i++){
        // rotate ray
        float angle = angles[i];
        float sx = rsx*cos(angle) - rsy*sin(angle) + 64 + 0.5f;
        float sy = rsx*sin(angle) + rsy*cos(angle) + 64 + 0.5f;
        float ex = rex*cos(angle) - rey*sin(angle) + 64 + 0.5f;
        float ey = rex*sin(angle) + rey*cos(angle) + 64 + 0.5f;
    
        float vx = (ex-sx)/128;
        float vy = (ey-sy)/128;

        float tmp = 0.0;
        for(int j = 0; j < 128; j++){
            tmp += tex2D<float>(texObj, sx+vx*j, sy+vy*j);
        }
        
        output[i*n_rays+ray_id] = tmp;
    }
}

void radon_forward_cuda(const float* x, const float* rays, const float* angles, float* y, const int n_rays, const int n_angles){
    cudaArray* tmp;
    auto my_tex = create_texture(x, tmp, 128, 128, 128);

    // Invoke kernel
    dim3 dimGrid(8);
    dim3 dimBlock(16);

    radon_forward_kernel<<<dimGrid, dimBlock>>>(y, my_tex, rays, angles, n_rays, n_angles);
}

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
