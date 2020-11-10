#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.h"
#include "texture.h"

template<bool parallel_beam, int channels, bool clip_to_circle, typename T>
__global__ void
radon_backward_kernel(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                      const RaysCfg cfg) {
    // Calculate image coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    const float cx = cfg.width / 2.0f;
    const float cy = cfg.height / 2.0f;
    const float cr = cfg.det_count / 2.0f;

    const float dx = float(x) - cx + 0.5f;
    const float dy = float(y) - cy + 0.5f;

    const float ids = __fdividef(1.0f, cfg.det_spacing);

    const uint batch_id = blockIdx.z * channels;

    __shared__ float s_sin[4096];
    __shared__ float s_cos[4096];

    for (int i = tid; i < cfg.n_angles; i += 256) {
        s_sin[i] = __sinf(angles[i]);
        s_cos[i] = __cosf(angles[i]);
    }
    __syncthreads();

    if (x < cfg.width && y < cfg.height) {

        float accumulator[channels];
#pragma unroll
        for (int i = 0; i < channels; i++) {
            accumulator[i] = 0.0f;
        }

        if (clip_to_circle) {
            const float r = hypot(dx, dy);
            if (r > cfg.det_count * 0.5f) {
                goto out;
            }
        }

        if (parallel_beam) {
            for (int i = 0; i < cfg.n_angles; i++) {
                float j = (s_cos[i] * dx + s_sin[i] * dy) * ids + cr;
                if (channels == 1) {
                    accumulator[0] += tex2DLayered<float>(texture, j, i + 0.5f, blockIdx.z);
                } else {
                    // read 4 values at the given position and accumulate
                    float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
                    accumulator[0] += read.x;
                    accumulator[1] += read.y;
                    accumulator[2] += read.z;
                    accumulator[3] += read.w;
                }
            }
        } else {
            const float kk = __fdividef(1.0f, cfg.s_dist + cfg.d_dist);

            for (int i = 0; i < cfg.n_angles; i++) {
                float den = kk * (-s_cos[i] * dy + s_sin[i] * dx + cfg.s_dist);
                float iden = __fdividef(1.0f, den);
                float j = (s_cos[i] * dx + s_sin[i] * dy) * ids * iden + cr;

                if (channels == 1) {
                    accumulator[0] += tex2DLayered<float>(texture, j, i + 0.5f, blockIdx.z) * iden;
                } else {
                    // read 4 values at the given position and accumulate
                    float4 read = tex2DLayered<float4>(texture, j, i + 0.5f, blockIdx.z);
                    accumulator[0] += read.x * iden;
                    accumulator[1] += read.y * iden;
                    accumulator[2] += read.z * iden;
                    accumulator[3] += read.w * iden;
                }
            }
        }

        out:
#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[(batch_id + b) * cfg.height * cfg.width + y * cfg.width + x] = accumulator[b] * ids;
        }
    }
}


template<typename T>
void radon_backward_cuda(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const RaysCfg &cfg, const int batch_size, const int device
) {
    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = (batch_size % 4 == 0) ? 4 : 1;

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get({device, batch_size, cfg.det_count, cfg.n_angles, channels, precision});
    tex->put(x);

    const int grid_size_h = roundup_div(cfg.height, 16);
    const int grid_size_w = roundup_div(cfg.width, 16);
    dim3 grid_dim(grid_size_h, grid_size_w, batch_size / channels);
    dim3 block_dim(16, 16);

    // Invoke kernel
    if (cfg.is_fanbeam) {
        if (channels == 1) {
            if (cfg.clip_to_circle) {
                radon_backward_kernel<false, 1, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else {
                radon_backward_kernel<false, 1, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            }
        } else {
            if (is_float) {
                if (cfg.clip_to_circle) {
                    radon_backward_kernel<false, 4, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                } else {
                    radon_backward_kernel<false, 4, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                }
            } else {
                if (cfg.clip_to_circle) {
                    radon_backward_kernel<false, 4, true> << < grid_dim, block_dim >> >
                    ((__half *) y, tex->texture, angles, cfg);
                } else {
                    radon_backward_kernel<false, 4, false> << < grid_dim, block_dim >> >
                    ((__half *) y, tex->texture, angles, cfg);
                }
            }
        }
    } else {
        if (channels == 1) {
            if (cfg.clip_to_circle) {
                radon_backward_kernel<true, 1, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            } else {
                radon_backward_kernel<true, 1, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
            }
        } else {
            if (is_float) {
                if (cfg.clip_to_circle) {
                    radon_backward_kernel<true, 4, true> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                } else {
                    radon_backward_kernel<true, 4, false> << < grid_dim, block_dim >> > (y, tex->texture, angles, cfg);
                }
            } else {
                if (cfg.clip_to_circle) {
                    radon_backward_kernel<true, 4, true> << < grid_dim, block_dim >> >
                    ((__half *) y, tex->texture, angles, cfg);
                } else {
                    radon_backward_kernel<true, 4, false> << < grid_dim, block_dim >> >
                    ((__half *) y, tex->texture, angles, cfg);
                }
            }
        }
    }
}

template void radon_backward_cuda<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache, const RaysCfg &cfg,
                                        const int batch_size, const int device);
template void radon_backward_cuda<unsigned short>(const unsigned short *x, const float *angles, unsigned short *y, TextureCache &tex_cache, const RaysCfg &cfg,
                                                 const int batch_size, const int device);