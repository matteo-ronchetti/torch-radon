#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <curand_kernel.h>
//#include <curand.h>
#include <cuda_fp16.h>


#include "utils.h"
#include "texture.h"

template<bool extend, int wpt>
__global__ void
radon_backward_kernel(float *output, cudaTextureObject_t texture, const float *rays, const float *angles,
                      const int img_size, const int n_rays, const int n_angles) {

    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    // Calculate image coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch_id = blockIdx.z * wpt;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    const float center = (img_size) / 2;
    const float max_r = center;
    float dx = (float) x - center + 0.5;
    float dy = (float) y - center + 0.5;

    float tmp[wpt];
#pragma unroll
    for (int i = 0; i < wpt; i++) tmp[i] = 0.0f;

    if (extend) {
//        if (r > max_r) {
//            dx *= max_r / r;
//            dy *= max_r / r;
//        }

        for (int i = 0; i < n_angles; i++) {
            float j = s_cos[i] * dx + s_sin[i] * dy + center;
#pragma unroll
            for (int b = 0; b < wpt; b++) {
                tmp[b] += tex2DLayered<float>(texture, j, i + 0.5f, batch_id + b);
            }
        }
    } else {
        const float r = hypot(dx, dy);
        if (r <= max_r) {
            for (int i = 0; i < n_angles; i++) {
                float j = s_cos[i] * dx + s_sin[i] * dy + center;
#pragma unroll
                for (int b = 0; b < wpt; b++) {
                    tmp[b] += tex2DLayered<float>(texture, j, i + 0.5f, batch_id + b);
                }
            }
        }
    }

#pragma unroll
    for (int b = 0; b < wpt; b++) {
        output[(batch_id + b) * img_size * img_size + y * img_size + x] = tmp[b];
    }
}

void radon_backward_cuda(const float *x, const float *rays, const float *angles, float *y, TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_rays, const int n_angles,
                         const int device, const bool extend) {
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, n_rays, n_angles, 1, PRECISION_FLOAT});
    tex->put(x);

    // if batch size is multiple of 4 each thread does 4 batches (is faster) (wpt = work per thread)
    const int wpt = (batch_size % 4 == 0) ? 4 : 1;
    const int grid_size = img_size / 16;
    dim3 dimGrid(grid_size, grid_size, batch_size / wpt);
    dim3 dimBlock(16, 16);

    // Invoke kernel
    if (extend) {
        if (wpt == 4) {
            radon_backward_kernel<true, 4> << < dimGrid, dimBlock >> >
                                                         (y, tex->texture, rays, angles, img_size, n_rays, n_angles);
        } else {
            radon_backward_kernel<true, 1> << < dimGrid, dimBlock >> >
                                                         (y, tex->texture, rays, angles, img_size, n_rays, n_angles);
        }
    } else {
        if (wpt == 4) {
            radon_backward_kernel<false, 4> << < dimGrid, dimBlock >> >
                                                          (y, tex->texture, rays, angles, img_size, n_rays, n_angles);
        } else {
            radon_backward_kernel<false, 1> << < dimGrid, dimBlock >> >
                                                          (y, tex->texture, rays, angles, img_size, n_rays, n_angles);
        }
    }
}


template<bool extend>
__global__ void
radon_backward_kernel_half(__half *output, cudaTextureObject_t texture, const float *rays, const float *angles,
                           const int img_size, const int n_rays, const int n_angles) {

    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    // Calculate image coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch_id = blockIdx.z * 4;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    const float center = (img_size) / 2;
    const float max_r = center;
    float dx = (float) x - center + 0.5;
    float dy = (float) y - center + 0.5;

    float tmp[4];
#pragma unroll
    for (int i = 0; i < 4; i++) tmp[i] = 0.0f;

    if (extend) {
        for (int i = 0; i < n_angles; i++) {
            float j = s_cos[i] * dx + s_sin[i] * dy + center;

            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x;
            tmp[1] += read.y;
            tmp[2] += read.z;
            tmp[3] += read.w;
        }
    } else {
        const float r = hypot(dx, dy);
        if (r <= max_r) {
            for (int i = 0; i < n_angles; i++) {
                float j = s_cos[i] * dx + s_sin[i] * dy + center;

                // read 4 values at the given position and accumulate
                float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
                tmp[0] += read.x;
                tmp[1] += read.y;
                tmp[2] += read.z;
                tmp[3] += read.w;
            }
        }
    }

#pragma unroll
    for (int b = 0; b < 4; b++) {
        output[(batch_id + b) * img_size * img_size + y * img_size + x] = __float2half(tmp[b]);
    }
}

void radon_backward_cuda(const unsigned short *x, const float *rays, const float *angles, unsigned short *y,
                         TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_rays, const int n_angles,
                         const int device, const bool extend) {
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, n_rays, n_angles, 4, PRECISION_HALF});
    tex->put(x);

    const int grid_size = img_size / 16;
    dim3 dimGrid(grid_size, grid_size, batch_size / 4);
    dim3 dimBlock(16, 16);

    // Invoke kernel
    if (extend) {
        radon_backward_kernel_half<true> << < dimGrid, dimBlock >> >
                                                       ((__half *) y, tex->texture, rays, angles, img_size, n_rays, n_angles);
    } else {
        radon_backward_kernel_half<false> << < dimGrid, dimBlock >> >
                                                        ((__half *) y, tex->texture, rays, angles, img_size, n_rays, n_angles);
    }
}


/*
template<typename T> __host__ __device__

inline T lerp(T v0, T v1, T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

template<bool extend, int wpt, int threads>
__global__ void radon_backward_kernel_lb(float *output, const float *sinogram, const float *rays, const float *angles,
                                         const int img_size, const int n_rays, const int n_angles,
                                         const int batch_size) {

    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    // Calculate image coordinates
    const uint batch_id = blockIdx.x * blockDim.x * wpt + threadIdx.x;
    const uint x = blockIdx.y * blockDim.y + threadIdx.y;
    const uint y = blockIdx.z;

    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    const float center = img_size / 2.0f - 0.5f;
    const float max_r = center;
    float dx = (float) x - center;
    float dy = (float) y - center;

    float tmp[wpt];
    for(int i = 0; i < wpt;i++) tmp[i] = 0.0f;
    const float r = hypot(dx, dy);

    for (int i = 0; i < n_angles; i++) {
        float j = s_cos[i] * dx + s_sin[i] * dy + center;
        float j_ceil = ceilf(j);
        float j_floor = j_ceil - 1.0f;
        float t = j - j_floor;
        const int base = i * img_size * batch_size + int(j_floor) * batch_size;
        if(j_floor >= 0 && j_ceil < img_size) {
            for(int b = 0; b < wpt; b++) {
                tmp[b] += lerp(sinogram[base + b*threads + batch_id],
                            sinogram[base + batch_size + b*threads + batch_id], t);

            }
        }
//        else{
//            if(j_floor < 0 && j_ceil >= 0) tmp[0] += lerp(0.0f, sinogram[base + batch_size + batch_id], t);
//            if(j_ceil >= img_size && j_floor < img_size) tmp[0] += lerp(sinogram[base + batch_id], 0.0f, t);
//        }
    }

    for(int b = 0; b < wpt; b++) {
        output[(batch_id + b*threads) * img_size * img_size + y * img_size + x] = tmp[b];
    }
}

void radon_backward_cuda_lb(const float *x, const float *rays, const float *angles, float *y, TextureCache &tex_cache,
                            const int batch_size, const int img_size, const int n_rays, const int n_angles,
                            const int device, const bool extend) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_backward_kernel_lb<true, 4, 32>, cudaFuncCachePreferL1));

    // Invoke kernel
    const int grid_size = img_size / 1;
    dim3 dimGrid(batch_size / (256), grid_size, img_size);
    dim3 dimBlock(256, 1);

    radon_backward_kernel_lb<true, 1, 128> << < dimGrid, dimBlock >> >
                                                        (y, x, rays, angles, img_size, n_rays, n_angles, batch_size);
//    if (extend) {
//        radon_backward_kernel_lb<true, 4, 32> << < dimGrid, dimBlock >> >
//                                                     (y, x, rays, angles, img_size, n_rays, n_angles, batch_size);
//    } else {
//        radon_backward_kernel_lb<false, 1, 32> << < dimGrid, dimBlock >> >
//                                                      (y, x, rays, angles, img_size, n_rays, n_angles, batch_size);
//    }
}
*/