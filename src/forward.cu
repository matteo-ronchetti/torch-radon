#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#include "utils.h"
#include "texture.h"

template<int angles_per_thread, int channels>
__global__ void
radon_forward_kernel(float *__restrict__ output, cudaTextureObject_t texture, const int det_count, const float det_spacing,
                     const float *__restrict__ angles,
                     const int img_size, const int n_angles) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = (blockIdx.y * blockDim.y + threadIdx.y) * angles_per_thread;
    const int batch_id = blockIdx.z * channels;

    if (angle_id < n_angles && ray_id < det_count) {
        // define registry caches
        float accumulator[angles_per_thread * channels];
        float2 s[angles_per_thread];
        float2 rv[angles_per_thread];

#pragma unroll
        for (int i = 0; i < angles_per_thread * channels; i++) {
            accumulator[i] = 0.0f;
        }

        const float v = (img_size) / 2.0;
        const float rsx = ray_id - v + 0.5f;
        const float rsy = 0.71f*img_size;
        const float rex = ray_id - v + 0.5f;
        const float rey = -rsy;

        const uint n_steps = __float2uint_ru(hypot(rex - rsx, rey - rsy));
        const float vx = (rex - rsx) / n_steps;
        const float vy = (rey - rsy) / n_steps;
        const float n = hypot(vx, vy);

        // rotate ray
#pragma unroll
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
#pragma unroll
            for (int i = 0; i < angles_per_thread; i++) {
                if (channels == 1) {
                    accumulator[i] += tex2DLayered<float>(texture, s[i].x, s[i].y, blockIdx.z);
                    s[i].x += rv[i].x;
                    s[i].y += rv[i].y;
                } else {
                    float4 read = tex2DLayered<float4>(texture, s[i].x, s[i].y, blockIdx.z);
                    accumulator[i * channels + 0] += read.x;
                    accumulator[i * channels + 1] += read.y;
                    accumulator[i * channels + 2] += read.z;
                    accumulator[i * channels + 3] += read.w;
                    s[i].x += rv[i].x;
                    s[i].y += rv[i].y;
                }
            }
        }

#pragma unroll
        for (int i = 0; i < angles_per_thread; i++) {
#pragma unroll
            for (int b = 0; b < channels; b++) {
                output[(batch_id + b) * det_count * n_angles + (angle_id + i) * det_count + ray_id] =
                        accumulator[i * channels + b] * n;
            }
        }
    }
}


void radon_forward_cuda(const float *x, const int det_count, const float det_spacing, const float *angles, float *y, TextureCache &tex_cache,
                        const int batch_size,
                        const int img_size, const int n_angles, const int device) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4, 1>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1, 1>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4, 4>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1, 4>, cudaFuncCachePreferL1));

    const int channels = (batch_size % 4 == 0) ? 4 : 1;
    const int angles_per_thread = (n_angles > 64) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, channels, PRECISION_FLOAT});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16 * angles_per_thread), batch_size / channels);


    if (n_angles <= 64) {
        if (channels == 1) {
            radon_forward_kernel<1, 1> << < grid_dim, block_dim >> >
                                                      (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_forward_kernel<1, 4> << < grid_dim, block_dim >> >
                                                      (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        }
    } else {
        if (channels == 1) {
            radon_forward_kernel<4, 1> << < grid_dim, block_dim >> >
                                                      (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_forward_kernel<4, 4> << < grid_dim, block_dim >> >
                                                      (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        }
    }
}

template<int angles_per_thread>
__global__ void
radon_forward_kernel_half(__half *__restrict__ output, cudaTextureObject_t texture, const int det_count, const float det_spacing,
                          const float *__restrict__ angles,
                          const int img_size, const int n_angles) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = (blockIdx.y * blockDim.y + threadIdx.y) * angles_per_thread;
    const int batch_id = blockIdx.z * 4;

    if (angle_id < n_angles && ray_id < det_count) {
        // define registry caches
        float accumulator[angles_per_thread * 4];
        float2 s[angles_per_thread];
        float2 rv[angles_per_thread];

#pragma unroll
        for (int i = 0; i < angles_per_thread * 4; i++) {
            accumulator[i] = 0.0f;
        }

        const float v = (img_size) / 2.0;
        const float rsx = ray_id - v + 0.5f;
        //const float radius = 0.71f*img_size;
        const float rsy = 0.71f*img_size; //sqrtf((radius + rsx)*(radius - rsx));
        const float rex = ray_id - v + 0.5f;
        const float rey = -rsy;

        const uint n_steps = __float2uint_ru(hypot(rex - rsx, rey - rsy));
        const float vx = (rex - rsx) / n_steps;
        const float vy = (rey - rsy) / n_steps;
        const float n = hypot(vx, vy);

        // rotate ray
#pragma unroll
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
#pragma unroll
            for (int i = 0; i < angles_per_thread; i++) {
                float4 read = tex2DLayered<float4>(texture, s[i].x, s[i].y, blockIdx.z);
                accumulator[i * 4 + 0] += read.x;
                accumulator[i * 4 + 1] += read.y;
                accumulator[i * 4 + 2] += read.z;
                accumulator[i * 4 + 3] += read.w;
                s[i].x += rv[i].x;
                s[i].y += rv[i].y;
            }
        }

#pragma unroll
        for (int i = 0; i < angles_per_thread; i++) {
#pragma unroll
            for (int b = 0; b < 4; b++) {
                output[(batch_id + b) * det_count * n_angles + (angle_id + i) * det_count + ray_id] =
                        accumulator[i * 4 + b] * n;
            }
        }
    }
}


void radon_forward_cuda(
        const unsigned short *x, const int det_count, const float det_spacing, const float *angles,
        unsigned short *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_angles, const int device
) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<4>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<1>, cudaFuncCachePreferL1));

    const int angles_per_thread = 1; //(n_angles > 64) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, 4, PRECISION_HALF});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16 * angles_per_thread), batch_size / 4);

//    if (n_angles <= 64) {
        radon_forward_kernel_half<1> << < grid_dim, block_dim >> >
                                               ((__half*)y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
//    } else {
//        radon_forward_kernel_half<4> << < grid_dim, block_dim >> >
//                                               ((__half*)y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
//    }
}