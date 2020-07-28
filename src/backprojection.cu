#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <curand_kernel.h>
//#include <curand.h>
#include <cuda_fp16.h>

#include "utils.h"
#include "texture.h"

template<int channels, bool clip_to_circle>
__global__ void
radon_backward_kernel(float *output, cudaTextureObject_t texture, const int det_count, const float det_spacing,
                      const float *angles,
                      const int img_size, const int n_angles) {

    // Calculate image coordinates
    
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const float center = img_size / 2.0f;
    const float dx = float(x) - center + 0.5f;
    const float dy = float(y) - center + 0.5f;

    const uint batch_id = blockIdx.z * channels;

    
    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    if (channels == 1) {
        float tmp = 0.0f;

        if(clip_to_circle){
            const float r = hypot(dx, dy);
            if (r <= center) {
                
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;
            tmp += tex2DLayered<float>(texture, j, i + 0.5f, batch_id);
        }

            }
        }else{
            
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;
            tmp += tex2DLayered<float>(texture, j, i + 0.5f, batch_id);
        }

        }

        output[batch_id * img_size * img_size + y * img_size + x] = tmp / det_spacing;
    } else {
        float tmp[channels];
#pragma unroll
        for (int i = 0; i < channels; i++) tmp[i] = 0.0f;

        if(clip_to_circle){
            const float r = hypot(dx, dy);
            if (r <= center) {
                
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;

            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x;
            tmp[1] += read.y;
            tmp[2] += read.z;
            tmp[3] += read.w;
        }
            }
        }else{
            
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;

            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x;
            tmp[1] += read.y;
            tmp[2] += read.z;
            tmp[3] += read.w;
        }
        }

#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[(batch_id + b) * img_size * img_size + y * img_size + x] = tmp[b] / det_spacing;
        }
    }
}


template<int wpt, bool clip_to_circle>
__global__ void
radon_backward_kernel_half(__half *output, cudaTextureObject_t texture, const int det_count, const float det_spacing,
                           const float *angles, const int img_size, const int n_angles) {
    // Calculate image coordinates
    
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const float center = img_size / 2.0f;
    const float dx = float(x) - center + 0.5f;
    const float dy = float(y) - center + 0.5f;

    const uint batch_id = blockIdx.z * 4 * wpt;

    
    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    float tmp[4 * wpt];
#pragma unroll
    for (int i = 0; i < 4 * wpt; i++) tmp[i] = 0.0f;

    if(clip_to_circle){
        const float r = hypot(dx, dy);
        if (r <= center) {
            
    const float ids = __fdividef(1.0f, det_spacing);
    for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;
#pragma unroll
        for (int h = 0; h < wpt; h++) {
            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z * wpt + h);
            tmp[h * 4 + 0] += read.x;
            tmp[h * 4 + 1] += read.y;
            tmp[h * 4 + 2] += read.z;
            tmp[h * 4 + 3] += read.w;
        }
    }
        }
    }else{
        
    const float ids = __fdividef(1.0f, det_spacing);
    for (int i = 0; i < n_angles; i++) {
            float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + center;
#pragma unroll
        for (int h = 0; h < wpt; h++) {
            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z * wpt + h);
            tmp[h * 4 + 0] += read.x;
            tmp[h * 4 + 1] += read.y;
            tmp[h * 4 + 2] += read.z;
            tmp[h * 4 + 3] += read.w;
        }
    }
    }

#pragma unroll
    for (int b = 0; b < 4 * wpt; b++) {
        output[(batch_id + b) * img_size * img_size + y * img_size + x] = __float2half(tmp[b] / det_spacing);
    }
}

void radon_backward_cuda(const unsigned short *x, const int det_count, const float det_spacing, const float *angles,
                         unsigned short *y,
                         TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_angles,
                         const int device, const bool clip_to_circle) {
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, det_count, n_angles, 4, PRECISION_HALF});
    tex->put(x);

    const int grid_size = img_size / 16;
    // performance gain obtained using wpt is is basically none
    // const int wpt = (batch_size % 16 == 0) ? 4 : 1;
    dim3 dimGrid(grid_size, grid_size, batch_size / (4));
    dim3 dimBlock(16, 16);

    // Invoke kernel
    if (clip_to_circle) {
        radon_backward_kernel_half<1, true> << < dimGrid, dimBlock >> >
                                                          ((__half *) y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
    } else {
        radon_backward_kernel_half<1, false> << < dimGrid, dimBlock >> >
                                                           ((__half *) y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);

    }

}

void radon_backward_cuda(const float *x, const int det_count, const float det_spacing, const float *angles, float *y,
                         TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_angles,
                         const int device, const bool clip_to_circle) {
    const int channels = (batch_size % 4 == 0) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, det_count, n_angles, channels, PRECISION_FLOAT});
    tex->put(x);

    // if batch size is multiple of 4 each thread does 4 batches (is faster) (wpt = work per thread)
    const int grid_size = img_size / 16;
    dim3 dimGrid(grid_size, grid_size, batch_size / channels);
    dim3 dimBlock(16, 16);

    // Invoke kernel
    if (clip_to_circle) {
        if (channels == 4) {
            radon_backward_kernel<4, true> << < dimGrid, dimBlock >> >
                                                         (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_backward_kernel<1, true> << < dimGrid, dimBlock >> >
                                                         (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        }
    } else {
        if (channels == 4) {
            radon_backward_kernel<4, false> << < dimGrid, dimBlock >> >
                                                          (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_backward_kernel<1, false> << < dimGrid, dimBlock >> >
                                                          (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        }
    }
}

template<int channels, bool clip_to_circle>
__global__ void
radon_backward_fanbeam_kernel(float *output, cudaTextureObject_t texture, const float s_dist, const float d_dist, const int det_count, const float det_spacing,
                      const float *angles,
                      const int img_size, const int n_angles) {

    // Calculate image coordinates
    
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const float center = img_size / 2.0f;
    const float dx = float(x) - center + 0.5f;
    const float dy = float(y) - center + 0.5f;

    const uint batch_id = blockIdx.z * channels;

    
    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    if (channels == 1) {
        float tmp = 0.0f;


        if(clip_to_circle){
            const float r = hypot(dx, dy);
            if (r <= center) {
                
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;

            tmp += tex2DLayered<float>(texture, j, i + 0.5f, batch_id) / den;
        }

            }
        }else{
            
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;

            tmp += tex2DLayered<float>(texture, j, i + 0.5f, batch_id) / den;
        }

        }

        output[batch_id * img_size * img_size + y * img_size + x] = tmp / det_spacing;
    } else {
        float tmp[channels];
#pragma unroll
        for (int i = 0; i < channels; i++) tmp[i] = 0.0f;

        if(clip_to_circle){
            const float r = hypot(dx, dy);
            if (r <= center) {
                
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;

            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x * iden;
            tmp[1] += read.y * iden;
            tmp[2] += read.z * iden;
            tmp[3] += read.w * iden;
        }
            }
        }else{
            
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;

            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
            tmp[0] += read.x * iden;
            tmp[1] += read.y * iden;
            tmp[2] += read.z * iden;
            tmp[3] += read.w * iden;
        }
        }

#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[(batch_id + b) * img_size * img_size + y * img_size + x] = tmp[b] / det_spacing;
        }
    }
}


template<int wpt, bool clip_to_circle>
__global__ void
radon_backward_fanbeam_kernel_half(__half *output, cudaTextureObject_t texture, const float s_dist, const float d_dist, const int det_count, const float det_spacing,
                           const float *angles, const int img_size, const int n_angles) {
    // Calculate image coordinates
    
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    const float center = img_size / 2.0f;
    const float dx = float(x) - center + 0.5f;
    const float dy = float(y) - center + 0.5f;

    const uint batch_id = blockIdx.z * 4 * wpt;

    
    __shared__ float s_sin[512];
    __shared__ float s_cos[512];

    for (int i = tid; i < n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    float tmp[4 * wpt];
#pragma unroll
    for (int i = 0; i < 4 * wpt; i++) tmp[i] = 0.0f;

    if(clip_to_circle){
        const float r = hypot(dx, dy);
        if (r <= center) {
            
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;
#pragma unroll
        for (int h = 0; h < wpt; h++) {
            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z * wpt + h);
            tmp[h * 4 + 0] += read.x * iden;
            tmp[h * 4 + 1] += read.y * iden;
            tmp[h * 4 + 2] += read.z * iden;
            tmp[h * 4 + 3] += read.w * iden;
        }
    }
        }
    }else{
        
        const float kk = __fdividef(1.0f, s_dist + d_dist);
        const float ids = __fdividef(1.0f, det_spacing);
        for (int i = 0; i < n_angles; i++) {
            float den = kk*(-s_cos[i] * dy + s_sin[i] * dx + s_dist);
            float iden = __fdividef(1.0f, den);
            float j = (s_cos[i] * dx + s_sin[i] * dy)*ids*iden + center;
#pragma unroll
        for (int h = 0; h < wpt; h++) {
            // read 4 values at the given position and accumulate
            float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z * wpt + h);
            tmp[h * 4 + 0] += read.x * iden;
            tmp[h * 4 + 1] += read.y * iden;
            tmp[h * 4 + 2] += read.z * iden;
            tmp[h * 4 + 3] += read.w * iden;
        }
    }
    }

#pragma unroll
    for (int b = 0; b < 4 * wpt; b++) {
        output[(batch_id + b) * img_size * img_size + y * img_size + x] = __float2half(tmp[b] / det_spacing);
    }
}

void radon_backward_fanbeam_cuda(const unsigned short *x, const float s_dist, const float d_dist, const int det_count, const float det_spacing, const float *angles,
                         unsigned short *y,
                         TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_angles,
                         const int device, const bool clip_to_circle) {
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, det_count, n_angles, 4, PRECISION_HALF});
    tex->put(x);

    const int grid_size = img_size / 16;
    // performance gain obtained using wpt is is basically none
    // const int wpt = (batch_size % 16 == 0) ? 4 : 1;
    dim3 dimGrid(grid_size, grid_size, batch_size / (4));
    dim3 dimBlock(16, 16);

    // Invoke kernel
    if (clip_to_circle) {
        radon_backward_fanbeam_kernel_half<1, true> << < dimGrid, dimBlock >> >
                                                          ((__half *) y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
    } else {
        radon_backward_fanbeam_kernel_half<1, false> << < dimGrid, dimBlock >> >
                                                           ((__half *) y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);

    }

}

void radon_backward_fanbeam_cuda(const float *x, const float s_dist, const float d_dist, const int det_count, const float det_spacing, const float *angles, float *y,
                         TextureCache &tex_cache,
                         const int batch_size, const int img_size, const int n_angles,
                         const int device, const bool clip_to_circle) {
    const int channels = (batch_size % 4 == 0) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, det_count, n_angles, channels, PRECISION_FLOAT});
    tex->put(x);

    // if batch size is multiple of 4 each thread does 4 batches (is faster) (wpt = work per thread)
    const int grid_size = img_size / 16;
    dim3 dimGrid(grid_size, grid_size, batch_size / channels);
    dim3 dimBlock(16, 16);

    // Invoke kernel
    if (clip_to_circle) {
        if (channels == 4) {
            radon_backward_fanbeam_kernel<4, true> << < dimGrid, dimBlock >> >
                                                         (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_backward_fanbeam_kernel<1, true> << < dimGrid, dimBlock >> >
                                                         (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        }
    } else {
        if (channels == 4) {
            radon_backward_fanbeam_kernel<4, false> << < dimGrid, dimBlock >> >
                                                          (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_backward_fanbeam_kernel<1, false> << < dimGrid, dimBlock >> >
                                                          (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        }
    }
}