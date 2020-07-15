#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#include "utils.h"
#include "texture.h"

template<int channels>
__global__ void
radon_forward_kernel(float *__restrict__ output, cudaTextureObject_t texture, const int det_count,
                     const float det_spacing,
                     const float *__restrict__ angles,
                     const int img_size, const int n_angles) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * channels;

    if (angle_id < n_angles && ray_id < det_count) {
        // define registry caches
        float accumulator[channels];

#pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }

        const float v = (img_size) / 2.0;

        // compute ray
        const float sx = ray_id - v + 0.5f;
        const float sy = 0.71f * img_size;
        const float ex = ray_id - v + 0.5f;
        const float ey = -sy;

        // rotate ray
        float angle = angles[angle_id];
        float cs = __cosf(angle);
        float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn + v;
        float rsy = sx * sn + sy * cs + v;
        float rdx = ex * cs - ey * sn + v - rsx;
        float rdy = ex * sn + ey * cs + v - rsy;

        // clip to square (to reduce memory reads)
        const float alpha_x_m = -rsx/rdx;
        const float alpha_x_p = (2*v - rsx)/rdx;
        const float alpha_y_m = -rsy/rdy;
        const float alpha_y_p = (2*v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        rsx += rdx*alpha_s;
        rsy += rdy*alpha_s;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        const uint n_steps = __float2uint_ru(hypot(rdx, rdy));
        const float vx = rdx / n_steps;
        const float vy = rdy / n_steps;
        const float n = hypot(vx, vy);

        for (uint j = 0; j <= n_steps; j++) { //changing j and n_steps to int makes everything way slower (WHY???)
            if (channels == 1) {
                accumulator[0] += tex2DLayered<float>(texture, rsx, rsy, blockIdx.z);
            } else {
                float4 read = tex2DLayered<float4>(texture, rsx, rsy, blockIdx.z);
                accumulator[0] += read.x;
                accumulator[1] += read.y;
                accumulator[2] += read.z;
                accumulator[3] += read.w;
            }
            rsx += vx;
            rsy += vy;
        }

#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] =
                    accumulator[b] * n;
        }
    }
}


void radon_forward_cuda(const float *x, const int det_count, const float det_spacing, const float *angles, float *y,
                        TextureCache &tex_cache,
                        const int batch_size,
                        const int img_size, const int n_angles, const int device) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4>, cudaFuncCachePreferL1));

    const int channels = (batch_size % 4 == 0) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, channels, PRECISION_FLOAT});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16), batch_size / channels);


    if (channels == 1) {
        radon_forward_kernel<1> << < grid_dim, block_dim >> >
                                                  (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
    } else {
        radon_forward_kernel<4> << < grid_dim, block_dim >> >
                                                  (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
    }
}

__global__ void
radon_forward_kernel_half(__half *__restrict__ output, cudaTextureObject_t texture, const int det_count,
                          const float det_spacing,
                          const float *__restrict__ angles,
                          const int img_size, const int n_angles) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * 4;

    if (angle_id < n_angles && ray_id < det_count) {
        // define registry caches
        float accumulator[4];

#pragma unroll
        for (int i = 0; i < 4; i++) {
            accumulator[i] = 0.0f;
        }

        const float v = img_size / 2.0;

        // compute ray
        const float sx = ray_id - v + 0.5f;
        const float sy = 0.71f * img_size;
//        const float ex = ray_id - v + 0.5f;
//        const float ey = -sy;

        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn + v;
        float rsy = sx * sn + sy * cs + v;
        float rdx = sx * cs + sy * sn + v - rsx;
        float rdy = sx * sn - sy * cs + v - rsy;

        // clip to square (to reduce memory reads)
        const float alpha_x_m = -rsx/rdx;
        const float alpha_x_p = (2*v - rsx)/rdx;
        const float alpha_y_m = -rsy/rdy;
        const float alpha_y_p = (2*v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        rsx += rdx*alpha_s;
        rsy += rdy*alpha_s;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        const uint n_steps = __float2uint_ru(hypot(rdx, rdy));
        const float vx = rdx / n_steps;
        const float vy = rdy / n_steps;
        const float n = hypot(vx, vy);


        for (uint j = 0; j <= n_steps; j++) { // changing j and n_steps to int makes everything way slower (WHY???)
            float4 read = tex2DLayered<float4>(texture, rsx, rsy, blockIdx.z);
            accumulator[0] += read.x;
            accumulator[1] += read.y;
            accumulator[2] += read.z;
            accumulator[3] += read.w;
            rsx += vx;
            rsy += vy;
        }

#pragma unroll
        for (int b = 0; b < 4; b++) {
            output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] =
                    accumulator[b] * n;
        }
    }
}


void radon_forward_cuda(
        const unsigned short *x, const int det_count, const float det_spacing, const float *angles,
        unsigned short *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_angles, const int device
) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half, cudaFuncCachePreferL1));

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, 4, PRECISION_HALF});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16), batch_size / 4);

    radon_forward_kernel_half << < grid_dim, block_dim >> >
                                             ((__half *) y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
}