#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void radon_forward_kernel(float* output, cudaTextureObject_t texObj, const float* rays, const float* angles, const int img_size, const int n_rays, const int n_angles) {
    // Calculate texture coordinates
    const uint ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint batch_id = blockIdx.y;
    const float rsx = rays[ray_id*4+0];
    const float rsy = rays[ray_id*4+1];
    const float rex = rays[ray_id*4+2];
    const float rey = rays[ray_id*4+3];

    for(int i = 0; i < n_angles; i++){
        // rotate ray
        float angle = angles[i];
        float sx = rsx*cos(angle) - rsy*sin(angle) + img_size/2;
        float sy = rsx*sin(angle) + rsy*cos(angle) + img_size/2;
        float ex = rex*cos(angle) - rey*sin(angle) + img_size/2;
        float ey = rex*sin(angle) + rey*cos(angle) + img_size/2;

        float vx = (ex-sx)/img_size;
        float vy = (ey-sy)/img_size;
        float n = hypot(vx, vy);

        float tmp = 0.0;
        for(int j = 0; j < img_size; j++){
            tmp += tex2DLayered<float>(texObj, sx+vx*j, sy+vy*j, batch_id);
        }

        output[batch_id*n_rays*n_angles + i*n_rays + ray_id] = tmp*n;
    }
}

void radon_forward_cuda(const float* x, const float* rays, const float* angles, float* y, const int batch_size, const int img_size, const int n_rays, const int n_angles){
    cudaArray* tmp;
    auto my_tex = create_texture(x, tmp, img_size, img_size, img_size);

    // Invoke kernel
    dim3 dimGrid(8, batch_size);
    dim3 dimBlock(16);

    radon_forward_kernel<<<dimGrid, dimBlock>>>(y, my_tex, rays, angles, img_size, n_rays, n_angles);

    cudaFreeArray(tmp);
}


int main(){
    std::cout << "Hello CUDA" << std::endl;
}
