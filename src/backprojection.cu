#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "utils.h"
#include "texture.h"

template<bool extend>
__global__ void radon_backward_kernel(float *output, cudaTextureObject_t texObj, const float *rays, const float *angles,
                                      const int img_size, const int n_rays, const int n_angles) {

    __shared__ float s_sin[256];
    __shared__ float s_cos[256];

    // Calculate image coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch_id = blockIdx.z;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < n_angles) {
        s_sin[tid] = __sinf(angles[tid]);
        s_cos[tid] = __cosf(angles[tid]);
    }
    __syncthreads();

    const float center = (img_size) / 2;
    const float max_r = center;
    float dx = (float) x - center + 0.5;
    float dy = (float) y - center + 0.5;

    float tmp = 0.0;
    const float r = hypot(dx, dy);

    if (extend) {
//        if (r > max_r) {
//            dx *= max_r / r;
//            dy *= max_r / r;
//        }

        for (int i = 0; i < n_angles; i++) {
            float j = s_cos[i] * dx + s_sin[i] * dy + center;
            tmp += tex2DLayered<float>(texObj, j, i + 0.5f, batch_id);
        }
    } else {
        if (r <= max_r) {
            for (int i = 0; i < n_angles; i++) {
                float j = s_cos[i] * dx + s_sin[i] * dy + center;
                tmp += tex2DLayered<float>(texObj, j, i + 0.5f, batch_id);
            }
        }
    }

    output[batch_id * img_size * img_size + y * img_size + x] = tmp;
}

void radon_backward_cuda(const float *x, const float *rays, const float *angles, float *y, TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_rays, const int n_angles,
                         const int device, const bool extend) {
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, n_rays, n_angles});
    tex->put(x);

    // Invoke kernel
    const int grid_size = img_size / 16;
    dim3 dimGrid(grid_size, grid_size, batch_size);
    dim3 dimBlock(16, 16);

    if (extend) {
        radon_backward_kernel<true> << < dimGrid, dimBlock >> >
                                                    (y, tex->texObj, rays, angles, img_size, n_rays, n_angles);
    } else {
        radon_backward_kernel<false> << < dimGrid, dimBlock >> >
                                                  (y, tex->texObj, rays, angles, img_size, n_rays, n_angles);
    }
}
