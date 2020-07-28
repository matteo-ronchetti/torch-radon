#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#include "utils.h"
#include "texture.h"

template<int channels, bool clip_to_circle>
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
        
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }


        // compute ray
        
        const float v = img_size / 2.0;
        const float sx = (ray_id - v + 0.5f)*det_spacing;
        const float sy = 0.71f * img_size;
        const float ex = sx;
        const float ey = -sy;


        
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;


        if(clip_to_circle){
            
        // clip rays to circle (to reduce memory reads)
        const float a = rdx * rdx + rdy * rdy;
        const float b = rsx * rdx + rsy * rdy;
        const float c = rsx * rsx + rsy * rsy - v * v;

        // min_clip to 1 to avoid getting empty rays
        const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
        const float alpha_s = (-b - delta_sqrt) / a;
        const float alpha_e = (-b + delta_sqrt) / a;
        
        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }else{
            
        // clip to square (to reduce memory reads)
        const float alpha_x_m = (-v - rsx)/rdx;
        const float alpha_x_p = (v - rsx)/rdx;
        const float alpha_y_m = (-v -rsy)/rdy;
        const float alpha_y_p = (v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        if(alpha_s > alpha_e){
            #pragma unroll
            for (int b = 0; b < channels; b++) {
                output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] = 0.0f;
            }
            return;
        }

        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }
        
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

template<bool clip_to_circle>
__global__ void
radon_forward_kernel_half(__half *__restrict__ output, cudaTextureObject_t texture, const int det_count,
                          const float det_spacing,
                          const float *__restrict__ angles,
                          const int img_size, const int n_angles) {
    constexpr int channels = 4;
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * 4;

    if (angle_id < n_angles && ray_id < det_count) {
        
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }


        // compute ray
        
        const float v = img_size / 2.0;
        const float sx = (ray_id - v + 0.5f)*det_spacing;
        const float sy = 0.71f * img_size;
        const float ex = sx;
        const float ey = -sy;


        
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;


        if(clip_to_circle){
            
        // clip rays to circle (to reduce memory reads)
        const float a = rdx * rdx + rdy * rdy;
        const float b = rsx * rdx + rsy * rdy;
        const float c = rsx * rsx + rsy * rsy - v * v;

        // min_clip to 1 to avoid getting empty rays
        const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
        const float alpha_s = (-b - delta_sqrt) / a;
        const float alpha_e = (-b + delta_sqrt) / a;
        
        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }else{
            
        // clip to square (to reduce memory reads)
        const float alpha_x_m = (-v - rsx)/rdx;
        const float alpha_x_p = (v - rsx)/rdx;
        const float alpha_y_m = (-v -rsy)/rdy;
        const float alpha_y_p = (v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        if(alpha_s > alpha_e){
            #pragma unroll
            for (int b = 0; b < channels; b++) {
                output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] = 0.0f;
            }
            return;
        }

        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }

        
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
                        const int img_size, const int n_angles, const int device, const bool clip_to_circle) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1, false>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4, false>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1, true>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4, true>, cudaFuncCachePreferL1));

    const int channels = (batch_size % 4 == 0) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, channels, PRECISION_FLOAT});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16), batch_size / channels);

    if(clip_to_circle){
        if (channels == 1) {
            radon_forward_kernel<1, true> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_forward_kernel<4, true> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        }
    }else{
        if (channels == 1) {
            radon_forward_kernel<1, false> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_forward_kernel<4, false> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
        }
    }
}


void radon_forward_cuda(
        const unsigned short *x, const int det_count, const float det_spacing, const float *angles,
        unsigned short *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_angles, const int device, const bool clip_to_circle
) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<true>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<false>, cudaFuncCachePreferL1));

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, 4, PRECISION_HALF});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16), batch_size / 4);

    if (clip_to_circle) {
        radon_forward_kernel_half < true > << <grid_dim, block_dim >> >
        ((__half *) y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
    } else {
        radon_forward_kernel_half < false > << <grid_dim, block_dim >> >
        ((__half *) y, tex->texture, det_count, det_spacing, angles, img_size, n_angles);
    }

}

template<int channels, bool clip_to_circle>
__global__ void
radon_forward_kernel_fanbeam(float *__restrict__ output, cudaTextureObject_t texture, const float s_dist, const float d_dist, const int det_count,
                     const float det_spacing,
                     const float *__restrict__ angles,
                     const int img_size, const int n_angles) {
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * channels;

    if (angle_id < n_angles && ray_id < det_count) {
        
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }


        // compute ray
        
        const float v = img_size / 2.0;
        const float sy = s_dist;
        const float sx = 0.0f;
        const float ey = -d_dist;
        const float ex = (ray_id - v + 0.5f)*det_spacing;


        
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;


        if(clip_to_circle){
            
        // clip rays to circle (to reduce memory reads)
        const float a = rdx * rdx + rdy * rdy;
        const float b = rsx * rdx + rsy * rdy;
        const float c = rsx * rsx + rsy * rsy - v * v;

        // min_clip to 1 to avoid getting empty rays
        const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
        const float alpha_s = (-b - delta_sqrt) / a;
        const float alpha_e = (-b + delta_sqrt) / a;
        
        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }else{
            
        // clip to square (to reduce memory reads)
        const float alpha_x_m = (-v - rsx)/rdx;
        const float alpha_x_p = (v - rsx)/rdx;
        const float alpha_y_m = (-v -rsy)/rdy;
        const float alpha_y_p = (v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        if(alpha_s > alpha_e){
            #pragma unroll
            for (int b = 0; b < channels; b++) {
                output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] = 0.0f;
            }
            return;
        }

        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }
        
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

template<bool clip_to_circle>
__global__ void
radon_forward_kernel_fanbeam_half(__half *__restrict__ output, cudaTextureObject_t texture, const float s_dist, const float d_dist, const int det_count,
                          const float det_spacing,
                          const float *__restrict__ angles,
                          const int img_size, const int n_angles) {
    constexpr int channels = 4;
    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_id = blockIdx.z * 4;

    if (angle_id < n_angles && ray_id < det_count) {
        
        float accumulator[channels];

        #pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }


        // compute ray
        
        const float v = img_size / 2.0;
        const float sy = s_dist;
        const float sx = 0.0f;
        const float ey = -d_dist;
        const float ex = (ray_id - v + 0.5f)*det_spacing;


        
        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd
        float rsx = sx * cs - sy * sn;
        float rsy = sx * sn + sy * cs;
        float rdx = ex * cs - ey * sn - rsx;
        float rdy = ex * sn + ey * cs - rsy;


        if(clip_to_circle){
            
        // clip rays to circle (to reduce memory reads)
        const float a = rdx * rdx + rdy * rdy;
        const float b = rsx * rdx + rsy * rdy;
        const float c = rsx * rsx + rsy * rsy - v * v;

        // min_clip to 1 to avoid getting empty rays
        const float delta_sqrt = sqrtf(max(b * b - a * c, 1.0f));
        const float alpha_s = (-b - delta_sqrt) / a;
        const float alpha_e = (-b + delta_sqrt) / a;
        
        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }else{
            
        // clip to square (to reduce memory reads)
        const float alpha_x_m = (-v - rsx)/rdx;
        const float alpha_x_p = (v - rsx)/rdx;
        const float alpha_y_m = (-v -rsy)/rdy;
        const float alpha_y_p = (v - rsy)/rdy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        if(alpha_s > alpha_e){
            #pragma unroll
            for (int b = 0; b < channels; b++) {
                output[(batch_id + b) * det_count * n_angles + angle_id * det_count + ray_id] = 0.0f;
            }
            return;
        }

        rsx += rdx*alpha_s + v;
        rsy += rdy*alpha_s + v;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        }

        
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


void radon_forward_fanbeam_cuda(const float *x, const float s_dist, const float d_dist, const int det_count, const float det_spacing, const float *angles, float *y,
                        TextureCache &tex_cache,
                        const int batch_size,
                        const int img_size, const int n_angles, const int device, const bool clip_to_circle) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1, false>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4, false>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<1, true>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel<4, true>, cudaFuncCachePreferL1));

    const int channels = (batch_size % 4 == 0) ? 4 : 1;
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, channels, PRECISION_FLOAT});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16), batch_size / channels);

    if(clip_to_circle){
        if (channels == 1) {
            radon_forward_kernel_fanbeam<1, true> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_forward_kernel_fanbeam<4, true> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        }
    }else{
        if (channels == 1) {
            radon_forward_kernel_fanbeam<1, false> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        } else {
            radon_forward_kernel_fanbeam<4, false> << < grid_dim, block_dim >> >
                                                   (y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
        }
    }
}


void radon_forward_fanbeam_cuda(
        const unsigned short *x, const float s_dist, const float d_dist,  const int det_count, const float det_spacing, const float *angles,
        unsigned short *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_angles, const int device, const bool clip_to_circle
) {
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<true>, cudaFuncCachePreferL1));
    checkCudaErrors(cudaFuncSetCacheConfig(radon_forward_kernel_half<false>, cudaFuncCachePreferL1));

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, img_size, img_size, 4, PRECISION_HALF});
    tex->put(x);

    // Invoke kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim(img_size / 16, roundup_div(n_angles, 16), batch_size / 4);

    if (clip_to_circle) {
        radon_forward_kernel_fanbeam_half < true > << <grid_dim, block_dim >> >
        ((__half *) y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
    } else {
        radon_forward_kernel_fanbeam_half < false > << <grid_dim, block_dim >> >
        ((__half *) y, tex->texture, s_dist, d_dist, det_count, det_spacing, angles, img_size, n_angles);
    }

}