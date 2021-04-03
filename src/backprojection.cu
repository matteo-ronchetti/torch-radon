#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.h"
#include "texture.h"
#include "backprojection.h"

template<bool parallel_beam, int channels, typename T>
__global__ void
radon_backward_kernel(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                      const VolumeCfg vol_cfg, const ProjectionCfg proj_cfg) {
    // Calculate image coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    const float cx = vol_cfg.width / 2.0f;
    const float cy = vol_cfg.height / 2.0f;
    const float cr = proj_cfg.det_count_u / 2.0f;

    const float dx = float(x) - cx + vol_cfg.dx + 0.5f;
    const float dy = float(y) - cy + vol_cfg.dy + 0.5f;

    const float ids = __fdividef(1.0f, proj_cfg.det_spacing_u);
    const float sdx = dx * ids;
    const float sdy = dy * ids;

//    const uint batch_id = blockIdx.z * channels;
    const int base = x + vol_cfg.width * (y + vol_cfg.height * blockIdx.z);
    const int pitch = vol_cfg.width * vol_cfg.height * blockDim.z * gridDim.z;

    __shared__ float2 sincos[4096];

    for (int i = tid; i < proj_cfg.n_angles; i += 256) {
        float2 tmp;
        tmp.x = __sinf(angles[i]);
        tmp.y = __cosf(angles[i]);
        sincos[i] = tmp;
    }
    __syncthreads();

    if (x < vol_cfg.width && y < vol_cfg.height) {
        float accumulator[channels];
#pragma unroll
        for (int i = 0; i < channels; i++) accumulator[i] = 0.0f;

        if (parallel_beam) {
            const int n_angles = proj_cfg.n_angles;
            
            // keep a float version of i to avoid expensive int2float conversions inside the main loop
            float fi = 0.5f;
            #pragma unroll(16)
            for (int i = 0; i < n_angles; i++) {
                float j = sincos[i].y * sdx + sincos[i].x * sdy + cr;
                if (channels == 1) {
                    accumulator[0] += tex2DLayered<float>(texture, j, fi, blockIdx.z);
                } else {
                    // read 4 values at the given position and accumulate
                    float4 read = tex2DLayered<float4>(texture, j, fi, blockIdx.z);
                    accumulator[0] += read.x;
                    accumulator[1] += read.y;
                    accumulator[2] += read.z;
                    accumulator[3] += read.w;
                }
                fi += 1.0f;
            }
        } else {
            const float k = proj_cfg.s_dist + proj_cfg.d_dist;
            const int n_angles = proj_cfg.n_angles;
            
            // keep a float version of i to avoid expensive int2float conversions inside the main loop
            float fi = 0.5f;
            #pragma unroll(16)
            for (int i = 0; i < n_angles; i++) {
                float iden;
                float den = fmaf(sincos[i].y, -dy, sincos[i].x * dx + proj_cfg.s_dist);
                
                // iden = __fdividef(k, den);
                asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(iden) : "f"(k), "f"(den));

                float j = (sincos[i].y * sdx + sincos[i].x * sdy) * iden + cr;

                if (channels == 1) {
                    accumulator[0] += tex2DLayered<float>(texture, j, fi, blockIdx.z) * iden;
                } else {
                    // read 4 values at the given position and accumulate
                    float4 read = tex2DLayered<float4>(texture, j, fi, blockIdx.z);
                    accumulator[0] += read.x * iden;
                    accumulator[1] += read.y * iden;
                    accumulator[2] += read.z * iden;
                    accumulator[3] += read.w * iden;
                }
                fi += 1.0f;
            }
        }

#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[base + b * pitch] = accumulator[b] * ids;
        }
    }
}


template<typename T>
void radon_backward_cuda(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const VolumeCfg &vol_cfg, const ProjectionCfg &proj_cfg, const ExecCfg &exec_cfg,
        const int batch_size, const int device
) {
    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = exec_cfg.get_channels(batch_size);

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    // const int tex_layers = proj_cfg.n_angles * batch_size / channels; 
    Texture *tex = tex_cache.get(
        {device, batch_size / channels, proj_cfg.n_angles, proj_cfg.det_count_u, true, channels, precision}
        // create_1Dlayered_texture_config(device, proj_cfg.det_count_u, tex_layers, channels, precision)
    );
    tex->put(x);

    // dim3 block_dim(16, 16);
    dim3 block_dim = exec_cfg.block_dim;
    dim3 grid_dim = exec_cfg.get_grid_size(vol_cfg.width, vol_cfg.height, batch_size / channels);

    // Invoke kernel
    if (proj_cfg.projection_type == FANBEAM) {
        if (channels == 1) {
            radon_backward_kernel<false, 1> << < grid_dim, block_dim >> >
                                                                      ((float*)y, tex->texture, angles, vol_cfg, proj_cfg);
        } else {
            if (is_float) {
                radon_backward_kernel<false, 4> << < grid_dim, block_dim >> >
                                                                          ((float*)y, tex->texture, angles, vol_cfg, proj_cfg);
            } else {
                radon_backward_kernel<false, 4> << < grid_dim, block_dim >> >
                                                                          ((__half *) y, tex->texture, angles, vol_cfg, proj_cfg);
            }
        }
    } else {
        if (channels == 1) {
            radon_backward_kernel<true, 1> << < grid_dim, block_dim >> >
                                                                     ((float*)y, tex->texture, angles, vol_cfg, proj_cfg);
        } else {
            if (is_float) {
                radon_backward_kernel<true, 4> << < grid_dim, block_dim >> >
                                                                         ((float*)y, tex->texture, angles, vol_cfg, proj_cfg);
            } else {
                radon_backward_kernel<true, 4> << < grid_dim, block_dim >> >
                                                                         ((__half *) y, tex->texture, angles, vol_cfg, proj_cfg);
            }
        }
    }
}

template void
radon_backward_cuda<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache,
                           const VolumeCfg &vol_cfg, const ProjectionCfg &proj_cfg, const ExecCfg &exec_cfg,
                           const int batch_size, const int device);

template void radon_backward_cuda<unsigned short>(const unsigned short *x, const float *angles, unsigned short *y,
                                                  TextureCache &tex_cache,
                                                  const VolumeCfg &vol_cfg, const ProjectionCfg &proj_cfg,
                                                  const ExecCfg &exec_cfg,
                                                  const int batch_size, const int device);


template<int channels, typename T>
__global__ void
radon_backward_kernel_3d(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                         const VolumeCfg vol_cfg, const ProjectionCfg proj_cfg) {
    // TODO consider det spacing both on U and V
    // Calculate volume coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint z = blockIdx.z * blockDim.z + threadIdx.z;
    const uint tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    const uint index = (z * vol_cfg.height + y) * vol_cfg.width + x;
    const uint pitch = vol_cfg.depth * vol_cfg.height * vol_cfg.width;

    const float cx = vol_cfg.width / 2.0f;
    const float cy = vol_cfg.height / 2.0f;
    const float cz = vol_cfg.depth / 2.0f;
    const float cu = proj_cfg.det_count_u / 2.0f;
    const float cv = proj_cfg.det_count_v / 2.0f;
    
    // TODO consider volume scale
    const float dx = float(x) - cx + 0.5f;
    const float dy = float(y) - cy + 0.5f;
    const float dz = float(z) - cz + 0.5f;

    const float inv_det_spacing = __fdividef(1.0f, proj_cfg.det_spacing_u);

    // TODO merge into a float2 array to save one memory load
    __shared__ float s_sin[4096];
    __shared__ float s_cos[4096];

    for (int i = tid; i < proj_cfg.n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    if (x < vol_cfg.width && y < vol_cfg.height && z < vol_cfg.depth) {
        float accumulator[channels];
#pragma unroll
        for (int i = 0; i < channels; i++) accumulator[i] = 0.0f;

        const float k = proj_cfg.s_dist + proj_cfg.d_dist;

        // TODO unroll
        for (int i = 0; i < proj_cfg.n_angles; i++) {
            // TODO consider det_spacing_v
            // TODO explicitly use FMA
            // TODO check PTX and optimize
            float k_over_alpha = __fdividef(k,
                                         (proj_cfg.s_dist + s_cos[i] * dy - s_sin[i] * dx) * proj_cfg.det_spacing_u);
            float beta = s_cos[i] * dx + s_sin[i] * dy;
            float u = k_over_alpha * beta;
            float v = k_over_alpha * dz;
            float scale = k_over_alpha * k_over_alpha;

            if (channels == 1) {
                accumulator[0] += tex2DLayered<float>(texture, u + cu, v + cv, i) * scale;
            } else {
                // read 4 values at the given position and accumulate
                float4 read = tex2DLayered<float4>(texture, u + cu, v + cv, i);
                accumulator[0] += read.x * scale;
                accumulator[1] += read.y * scale;
                accumulator[2] += read.z * scale;
                accumulator[3] += read.w * scale;
            }
        }

#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[b * pitch + index] = accumulator[b];
        }
    }
}


template<typename T>
void radon_backward_cuda_3d(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const VolumeCfg &vol_cfg, const ProjectionCfg &proj_cfg, const ExecCfg &exec_cfg, const int batch_size,
        const int device
) {
    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = exec_cfg.get_channels(batch_size);

    Texture *tex = tex_cache.get(
            {device, proj_cfg.n_angles, proj_cfg.det_count_v, proj_cfg.det_count_u, true, channels, precision});

    dim3 grid_dim = exec_cfg.get_grid_size(vol_cfg.width, vol_cfg.height, vol_cfg.depth);

    for (int i = 0; i < batch_size; i += channels) {
        T *local_y = &y[i * vol_cfg.depth * vol_cfg.height * vol_cfg.width];
        tex->put(&x[i * proj_cfg.n_angles * proj_cfg.det_count_v * proj_cfg.det_count_u]);

        // Invoke kernel
        if (channels == 1) {
            radon_backward_kernel_3d<1> << < grid_dim, exec_cfg.block_dim >> >
                                                       (local_y, tex->texture, angles, vol_cfg, proj_cfg);
        } else {
            if (is_float) {
                radon_backward_kernel_3d<4> << < grid_dim, exec_cfg.block_dim >> >
                                                           (local_y, tex->texture, angles, vol_cfg, proj_cfg);
            } else {
                radon_backward_kernel_3d<4> << < grid_dim, exec_cfg.block_dim >> >
                                                           ((__half *) local_y, tex->texture, angles, vol_cfg, proj_cfg);
            }
        }
    }
}

template void radon_backward_cuda_3d<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache,
                                            const VolumeCfg &vol_cfg, const ProjectionCfg &proj_cfg,
                                            const ExecCfg &exec_cfg,
                                            const int batch_size, const int device);

template void radon_backward_cuda_3d<unsigned short>(const unsigned short *x, const float *angles, unsigned short *y,
                                                     TextureCache &tex_cache,
                                                     const VolumeCfg &vol_cfg, const ProjectionCfg &proj_cfg,
                                                     const ExecCfg &exec_cfg,
                                                     const int batch_size, const int device);