#include <iostream>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "utils.h"
#include "texture.h"

template<int angles_per_thread>
__global__ void
radon_forward_kernel(float *__restrict__ output, cudaTextureObject_t texObj, const float *__restrict__ rays,
                     const float *__restrict__ angles,
                     const int img_size, const int n_rays, const int n_angles) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = (blockIdx.y * blockDim.y + threadIdx.y) * angles_per_thread;
    const int batch_id = blockIdx.z;

    if (angle_id < n_angles) {
        // define registry caches
        float accumulator[angles_per_thread];
        float2 s[angles_per_thread];
        //float sy[angles_per_thread];
        float2 rv[angles_per_thread];
        //float rvy[angles_per_thread];

        for (int i = 0; i < angles_per_thread; i++) {
            accumulator[i] = 0.0f;
        }

        const float rsx = rays[ray_id * 4 + 0];
        const float rsy = rays[ray_id * 4 + 1];
        const float rex = rays[ray_id * 4 + 2];
        const float rey = rays[ray_id * 4 + 3];
        const float v = (img_size) / 2.0; //

        const uint n_steps = __float2uint_ru(hypot(rex - rsx, rey - rsy)); //
        const float vx = (rex - rsx) / n_steps; //
        const float vy = (rey - rsy) / n_steps; //
        const float n = hypot(vx, vy); //

        // rotate ray
        for (int i = 0; i < angles_per_thread; i++) {
            float angle = angles[angle_id + i];
            float cs = __cosf(angle);
            float sn = __sinf(angle);

            s[i].x = rsx * cs - rsy * sn + v;
            s[i].y = rsx * sn + rsy * cs + v;
            rv[i].x = vx * cs - vy * sn;
            rv[i].y = vx * sn + vy * cs;
        }

        for (uint j = 0; j <= n_steps; j++) { //changing j and n_steps to int makes everything way slower (WHY???)
            for (int i = 0; i < angles_per_thread; i++) {
                accumulator[i] += tex2DLayered<float>(texObj, s[i].x, s[i].y, batch_id);
                s[i].x += rv[i].x;
                s[i].y += rv[i].y;
            }
        }

        for (int i = 0; i < angles_per_thread; i++) {
            output[batch_id * n_rays * n_angles + (angle_id + i) * n_rays + ray_id] =
                    accumulator[i] * n;
        }
    }
}


void radon_forward_cuda(const float *x, const float *rays, const float *angles, float *y, TextureCache &tex_cache,
                        const int batch_size,
                        const int img_size, const int n_rays, const int n_angles, const int device) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1>, cudaFuncCachePreferL1));

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture* tex = tex_cache.get({device, batch_size, img_size, img_size});
    tex->put(x);

    // Invoke kernel
    const int grid_size = img_size / 16;
    dim3 dimBlock(16, 16);

    if (n_angles <= 64) {
        dim3 dimGrid(grid_size, roundup_div(n_angles, 16), batch_size);
        radon_forward_kernel<1> << < dimGrid, dimBlock >> >
                                              (y, tex->texObj, rays, angles, img_size, n_rays, n_angles);
    } else {
        dim3 dimGrid(grid_size, roundup_div(n_angles, 16*4), batch_size);
        radon_forward_kernel<4> << < dimGrid, dimBlock >> >
                                              (y, tex->texObj, rays, angles, img_size, n_rays, n_angles);
    }
}