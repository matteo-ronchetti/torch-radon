#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.h"
#include "texture.h"
#include "parameter_classes.h"
#include "log.h"
#include "vectors.cuh"


template<bool parallel_beam, int channels, typename T>
__global__ void
radon_forward_kernel(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                     const VolumeCfg vol_cfg, const Projection2D proj_cfg) {

    // Calculate texture coordinates
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;

    const int base = ray_id + proj_cfg.det_count * (angle_id + proj_cfg.n_angles * blockIdx.z);
    const int mem_pitch = proj_cfg.det_count * proj_cfg.n_angles * blockDim.z * gridDim.z;

    if (angle_id < proj_cfg.n_angles && ray_id < proj_cfg.det_count) {
        float accumulator[channels];

#pragma unroll
        for (int i = 0; i < channels; i++) accumulator[i] = 0.0f;

        // compute ray
        float sx, sy, ex, ey;
        if (parallel_beam) {
            sx = (ray_id - proj_cfg.det_count * 0.5f + 0.5f) * proj_cfg.det_spacing;
            sy = hypot(abs(proj_cfg.dx) + vol_cfg.width * 0.5f, abs(proj_cfg.dy) + vol_cfg.height * 0.5f);
            ex = sx;
            ey = -sy;
        } else {
            sy = proj_cfg.s_dist;
            sx = 0.0f;
            ey = -proj_cfg.d_dist;
            ex = (ray_id - proj_cfg.det_count * 0.5f + 0.5f) * proj_cfg.det_spacing;
        }

        // rotate ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        // start position rs and direction rd (in detector coordinate system)
        float rsx = sx * cs + sy * sn;
        float rsy = -sx * sn + sy * cs;
        float rdx = ex * cs + ey * sn - rsx;
        float rdy = -ex * sn + ey * cs - rsy;

        // convert coordinates to volume coordinate system
        const float vol_orig_x = proj_cfg.dx - 0.5f * vol_cfg.width * vol_cfg.spacing.x;
        const float vol_orig_y = proj_cfg.dy - 0.5f * vol_cfg.height * vol_cfg.spacing.y;
        rsx = (rsx - vol_orig_x) * vol_cfg.inv_spacing.x;
        rsy = (rsy - vol_orig_y) * vol_cfg.inv_spacing.y;
        rdx = rdx * vol_cfg.inv_spacing.x;
        rdy = rdy * vol_cfg.inv_spacing.y;


        // clip to volume (to reduce memory reads)
        float dx = rdx >= 0 ? max(rdx, 1e-6) : min(rdx, -1e-6);
        float dy = rdy >= 0 ? max(rdy, 1e-6) : min(rdy, -1e-6);

        const float alpha_x_m = (-rsx) / dx;
        const float alpha_x_p = (vol_cfg.width - rsx) / dx;
        const float alpha_y_m = (-rsy) / dy;
        const float alpha_y_p = (vol_cfg.height - rsy) / dy;
        const float alpha_s = max(min(alpha_x_p, alpha_x_m), min(alpha_y_p, alpha_y_m));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), max(alpha_y_p, alpha_y_m));

        // if ray volume intersection is empty exit
        if (alpha_s > alpha_e - 1e-6) {
#pragma unroll
            for (int b = 0; b < channels; b++) output[base + b * mem_pitch] = 0.0f;
            return;
        }

        rsx += rdx * alpha_s;
        rsy += rdy * alpha_s;
        rdx *= (alpha_e - alpha_s);
        rdy *= (alpha_e - alpha_s);

        const int n_steps = __float2int_rn(max(abs(rdx), abs(rdy)));
        const float vx = rdx / max(abs(rdx), abs(rdy));
        const float vy = rdy / max(abs(rdx), abs(rdy));
        const float n = hypot(vx * vol_cfg.spacing.x, vy * vol_cfg.spacing.y);

        float step;
        if (abs(rdy) >= abs(rdx)) {
            float y_increment = 0.5f - rsy + __float2int_rn(rsy);
            step = y_increment / vy;
            step += vy < 0;
        } else {
            float x_increment = 0.5f - rsx + __float2int_rn(rsx);
            step = x_increment / vx;
            step += vx < 0;
        }
        rsx += step * vx;
        rsy += step * vy;

#pragma unroll(4)
        for (int j = 0; j < n_steps; j++) {
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
        for (int b = 0; b < channels; b++) output[base + b * mem_pitch] = accumulator[b] * n;
    }
}

template<typename T>
void radon_forward_cuda(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const VolumeCfg &vol_cfg, const Projection2D &proj_cfg, const ExecCfg &exec_cfg,
        const int batch_size, const int device
) {
    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = exec_cfg.get_channels(batch_size);

    LOG_DEBUG("Radon forward 2D. Height: " << vol_cfg.height << " width: " << vol_cfg.width << " channels: " << channels);
    LOG_DEBUG("Radon forward 2D. Det count: " << proj_cfg.det_count << " angles: " << proj_cfg.n_angles << " batch_size: " << batch_size);

    // copy x into CUDA Array (allocating it if needed) and bind to texture
    Texture *tex = tex_cache.get(
            {device, batch_size / channels, vol_cfg.height, vol_cfg.width, true, channels, precision});
    tex->put(x);

    // Invoke kernel
    const dim3 grid_dim = exec_cfg.get_grid_size(proj_cfg.det_count, proj_cfg.n_angles, batch_size / channels);
    const dim3 block_dim = exec_cfg.get_block_dim();

    LOG_DEBUG("Block Size x:" << block_dim.x << " y:" << block_dim.y << " z:" << block_dim.z);
    LOG_DEBUG("Grid Size x:" << grid_dim.x << " y:" << grid_dim.y << " z:" << grid_dim.z);

    if (proj_cfg.type == ProjectionType::FanBeam) {
        if (channels == 1) {
            radon_forward_kernel<false, 1> << < grid_dim, block_dim >> >
            ((float *) y, tex->texture, angles, vol_cfg, proj_cfg);
        } else {
            if (is_float) {
                radon_forward_kernel<false, 4> << < grid_dim, block_dim >> >
                ((float *) y, tex->texture, angles, vol_cfg, proj_cfg);
            } else {
                radon_forward_kernel<false, 4> << < grid_dim, block_dim >> >
                ((__half *) y, tex->texture, angles, vol_cfg, proj_cfg);
            }
        }
    } else {
        if (channels == 1) {
            radon_forward_kernel<true, 1> << < grid_dim, block_dim >> >
            ((float *) y, tex->texture, angles, vol_cfg, proj_cfg);
        } else {
            if (is_float) {
                radon_forward_kernel<true, 4> << < grid_dim, block_dim >> >
                ((float *) y, tex->texture, angles, vol_cfg, proj_cfg);
            } else {
                radon_forward_kernel<true, 4> << < grid_dim, block_dim >> >
                ((__half *) y, tex->texture, angles, vol_cfg, proj_cfg);
            }
        }
    }
}

template void
radon_forward_cuda<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache,
                          const VolumeCfg &vol_cfg, const Projection2D &proj_cfg, const ExecCfg &exec_cfg,
                          const int batch_size, const int device);

template void radon_forward_cuda<ushort>(const ushort *x, const float *angles, ushort *y,
                                         TextureCache &tex_cache,
                                         const VolumeCfg &vol_cfg, const Projection2D &proj_cfg,
                                         const ExecCfg &exec_cfg,
                                         const int batch_size, const int device);


template<int channels, typename T>
__global__ void
radon_forward_kernel_3d(T *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                        const VolumeCfg vol_cfg, const Projection3D proj_cfg) {
    // Calculate sensor coordinates in pixels
    // TODO is there an "optimal" map from thread to coordinates that maximizes cache hits?
    // TODO check other permutations (combined with different block sizes)
    const int pu = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int pv = blockIdx.z * blockDim.z + threadIdx.z;

    const uint index = (angle_id * proj_cfg.det_count_v + pv) * proj_cfg.det_count_u + pu;
    const uint mem_pitch = proj_cfg.n_angles * proj_cfg.det_count_v * proj_cfg.det_count_u;

    if (angle_id < proj_cfg.n_angles && pu < proj_cfg.det_count_u && pv < proj_cfg.det_count_v) {
        // define accumulator
        float accumulator[channels];
#pragma unroll
        for (int i = 0; i < channels; i++) accumulator[i] = 0.0f;

        // compute ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        vec3 source = {0.0, -proj_cfg.s_dist, 0.0};
        float u = (pu - proj_cfg.det_count_u * 0.5f + 0.5f);
        float v = (pv - proj_cfg.det_count_v * 0.5f + 0.5f);
        vec3 dir = {u * proj_cfg.det_spacing_u, proj_cfg.s_dist + proj_cfg.d_dist, v * proj_cfg.det_spacing_v};

        // rotate start/end positions and add pitch * angle / (2*pi) to z
        source = rotxy_transz(source, sn, cs, proj_cfg.pitch * angle * 0.1591549f);
        dir = rotxy(dir, sn, cs);

        // convert coordinates to volume coordinate system
        source = proj_cfg.worldToVoxel * source;
        dir = rotate_scale(proj_cfg.worldToVoxel, dir);

        // Clip ray to cube to reduce the number of memory reads
        float dx = dir.x >= 0 ? max(dir.x, 1e-6) : min(dir.x, -1e-6);
        float dy = dir.y >= 0 ? max(dir.y, 1e-6) : min(dir.y, -1e-6);
        float dz = dir.z >= 0 ? max(dir.z, 1e-6) : min(dir.z, -1e-6);

        const float alpha_x_m = (-source.x) / dx;
        const float alpha_x_p = (vol_cfg.width - source.x) / dx;
        const float alpha_y_m = (-source.y) / dy;
        const float alpha_y_p = (vol_cfg.height - source.y) / dy;
        const float alpha_z_m = (-source.z) / dz;
        const float alpha_z_p = (vol_cfg.slices - source.z) / dz;

        const float alpha_s = max(min(alpha_x_p, alpha_x_m), max(min(alpha_y_p, alpha_y_m), min(alpha_z_p, alpha_z_m)));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), min(max(alpha_y_p, alpha_y_m), max(alpha_z_p, alpha_z_m)));

        if (alpha_s > alpha_e - 1e-6) {
#pragma unroll
            for (int b = 0; b < channels; b++) output[b * mem_pitch + index] = 0.0f;
            return;
        }

        source = source + dir * alpha_s;
        dir = dir * (alpha_e - alpha_s);

        // Compute number of steps
        const float f_n_steps = max(abs(dir.x), max(abs(dir.y), abs(dir.z)));
        const int n_steps = __float2uint_ru(f_n_steps);
        vec3 w = dir * __fdividef(1.0f, f_n_steps);
        const float n = norm(w * vol_cfg.spacing);

        // Do a small step to start near the middle of the first voxel (?)
        float step;
        if (abs(dir.y) >= abs(dir.x)) {
            float y_increment = 0.5f - source.y + __float2int_rn(source.y);
            step = y_increment / w.y;
            step += w.y < 0;
        } else {
            float x_increment = 0.5f - source.x + __float2int_rn(source.x);
            step = x_increment / w.x;
            step += w.x < 0;
        }
        source = source + step * w;

        // accumulate loop
#pragma unroll(4)
        for (int j = 0; j <= n_steps; j++) {
            if (channels == 1) {
                accumulator[0] += tex3D<float>(texture, source.x, source.y, source.z);
            } else {
                float4 read = tex3D<float4>(texture, source.x, source.y, source.z);
                accumulator[0] += read.x;
                accumulator[1] += read.y;
                accumulator[2] += read.z;
                accumulator[3] += read.w;
            }

            source += w;
        }

        // output
#pragma unroll
        for (int b = 0; b < channels; b++) {
            output[b * mem_pitch + index] = accumulator[b] * n;
        }
    }
}

template<typename T>
void radon_forward_cuda_3d(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const VolumeCfg &vol_cfg, Projection3D &proj_cfg, const ExecCfg &exec_cfg,
        const int batch_size, const int device
) {
    constexpr bool is_float = std::is_same<T, float>::value;
    constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;
    const int channels = exec_cfg.get_channels(batch_size);

    Texture *tex = tex_cache.get(
            {device, vol_cfg.slices, vol_cfg.height, vol_cfg.width, false, channels, precision});

    const dim3 grid_dim = exec_cfg.get_grid_size(proj_cfg.det_count_u, proj_cfg.n_angles, proj_cfg.det_count_v);
    const dim3 block_dim = exec_cfg.get_block_dim();

    proj_cfg.updateMatrices(vol_cfg);
    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < batch_size; i += channels) {
        T *local_y = &y[i * proj_cfg.det_count_u * proj_cfg.det_count_v * proj_cfg.n_angles];
        tex->put(&x[i * vol_cfg.slices * vol_cfg.height * vol_cfg.width]);
        checkCudaErrors(cudaDeviceSynchronize());

        // Invoke kernel
        if (channels == 1) {
            radon_forward_kernel_3d<1> << < grid_dim, block_dim >> >
            (local_y, tex->texture, angles, vol_cfg, proj_cfg);
            checkCudaErrors(cudaDeviceSynchronize());
        } else {
            if (is_float) {
                radon_forward_kernel_3d<4> << < grid_dim, block_dim >> >
                (local_y, tex->texture, angles, vol_cfg, proj_cfg);
            } else {
                radon_forward_kernel_3d<4> << < grid_dim, block_dim >> >
                ((__half *) local_y, tex->texture, angles, vol_cfg, proj_cfg);
            }
        }
    }
}

template void
radon_forward_cuda_3d<float>(const float *x, const float *angles, float *y, TextureCache &tex_cache,
                             const VolumeCfg &vol_cfg, Projection3D &proj_cfg, const ExecCfg &exec_cfg,
                             const int batch_size, const int device);

template void radon_forward_cuda_3d<ushort>(const ushort *x, const float *angles, ushort *y,
                                            TextureCache &tex_cache,
                                            const VolumeCfg &vol_cfg, Projection3D &proj_cfg,
                                            const ExecCfg &exec_cfg,
                                            const int batch_size, const int device);


__global__ void radon_forward_kernel_3d_batch(float *__restrict__ output, cudaTextureObject_t texture, const float *__restrict__ angles,
                                              const VolumeCfg vol_cfg, const Projection3D *__restrict__ proj_cfgs, int n_angles) {
    // Calculate sensor coordinates in pixels
    const int pu = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int pv = blockIdx.z * blockDim.z + threadIdx.z;

    const uint index = (angle_id * proj_cfgs[0].det_count_v + pv) * proj_cfgs[0].det_count_u + pu;

    if (angle_id < n_angles && pu < proj_cfgs[0].det_count_u && pv < proj_cfgs[0].det_count_v) {
        const Projection3D proj_cfg = proj_cfgs[angle_id];

        // define accumulator
        float accumulator = 0.0f;

        // compute ray
        const float angle = angles[angle_id];
        const float cs = __cosf(angle);
        const float sn = __sinf(angle);

        vec3 source = {0.0, -proj_cfg.s_dist, 0.0};
        float u = (pu - proj_cfg.det_count_u * 0.5f + 0.5f);
        float v = (pv - proj_cfg.det_count_v * 0.5f + 0.5f);
        vec3 dir = {u * proj_cfg.det_spacing_u, proj_cfg.s_dist + proj_cfg.d_dist, v * proj_cfg.det_spacing_v};

        // rotate start/end positions and add pitch * angle / (2*pi) to z
        source = rotxy_transz(source, sn, cs, proj_cfg.pitch * angle * 0.1591549f);
        dir = rotxy(dir, sn, cs);

        // convert coordinates to volume coordinate system
        source = proj_cfg.worldToVoxel * source;
        dir = rotate_scale(proj_cfg.worldToVoxel, dir);

        // Clip ray to cube to reduce the number of memory reads
        float dx = dir.x >= 0 ? max(dir.x, 1e-6) : min(dir.x, -1e-6);
        float dy = dir.y >= 0 ? max(dir.y, 1e-6) : min(dir.y, -1e-6);
        float dz = dir.z >= 0 ? max(dir.z, 1e-6) : min(dir.z, -1e-6);

        const float alpha_x_m = (-source.x) / dx;
        const float alpha_x_p = (vol_cfg.width - source.x) / dx;
        const float alpha_y_m = (-source.y) / dy;
        const float alpha_y_p = (vol_cfg.height - source.y) / dy;
        const float alpha_z_m = (-source.z) / dz;
        const float alpha_z_p = (vol_cfg.slices - source.z) / dz;

        const float alpha_s = max(min(alpha_x_p, alpha_x_m), max(min(alpha_y_p, alpha_y_m), min(alpha_z_p, alpha_z_m)));
        const float alpha_e = min(max(alpha_x_p, alpha_x_m), min(max(alpha_y_p, alpha_y_m), max(alpha_z_p, alpha_z_m)));

        if (alpha_s > alpha_e - 1e-6) {
            output[index] = 0.0f;
            return;
        }

        source = source + dir * alpha_s;
        dir = dir * (alpha_e - alpha_s);

        // Compute number of steps
        const float f_n_steps = max(abs(dir.x), max(abs(dir.y), abs(dir.z)));
        const int n_steps = __float2uint_ru(f_n_steps);
        vec3 w = dir * __fdividef(1.0f, f_n_steps);
        const float n = norm(w * vol_cfg.spacing);

        // Do a small step to start near the middle of the first voxel (?)
        float step;
        if (abs(dir.y) >= abs(dir.x)) {
            float y_increment = 0.5f - source.y + __float2int_rn(source.y);
            step = y_increment / w.y;
            step += w.y < 0;
        } else {
            float x_increment = 0.5f - source.x + __float2int_rn(source.x);
            step = x_increment / w.x;
            step += w.x < 0;
        }
        source = source + step * w;

        // accumulate loop
#pragma unroll(4)
        for (int j = 0; j <= n_steps; j++) {
            accumulator += tex3D<float>(texture, source.x, source.y, source.z);
            source += w;
        }

        output[index] = accumulator;
    }
}

void radon_forward_cuda_3d_batch(
        const float *x, const float *angles, float *y, TextureCache &tex_cache,
        const VolumeCfg &vol_cfg, std::vector <Projection3D> &proj_cfgs, const ExecCfg &exec_cfg,
        const int batch_size, const int device
) {
    Texture *tex = tex_cache.get({device, vol_cfg.slices, vol_cfg.height, vol_cfg.width, false, 1, PRECISION_FLOAT});

    const dim3 grid_dim = exec_cfg.get_grid_size(proj_cfgs[0].det_count_u, proj_cfgs.size(), proj_cfgs[0].det_count_v);
    const dim3 block_dim = exec_cfg.get_block_dim();

    for (auto &proj_cfg: proj_cfgs)
        proj_cfg.updateMatrices(vol_cfg);

    Projection3D *cuda_proj_cfgs = nullptr;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **) &cuda_proj_cfgs, proj_cfgs.size() * sizeof(Projection3D)));
    checkCudaErrors(cudaMemcpy(cuda_proj_cfgs, &proj_cfgs[0], proj_cfgs.size() * sizeof(Projection3D), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < batch_size; i++) {
        float *local_y = &y[i * proj_cfgs[0].det_count_u * proj_cfgs[0].det_count_v * int(proj_cfgs.size())];
        tex->put(&x[i * vol_cfg.slices * vol_cfg.height * vol_cfg.width]);
        checkCudaErrors(cudaDeviceSynchronize());

        radon_forward_kernel_3d_batch<<< grid_dim, block_dim >>>(local_y, tex->texture, angles, vol_cfg, cuda_proj_cfgs, int(proj_cfgs.size()));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaFree(cuda_proj_cfgs));
    checkCudaErrors(cudaDeviceSynchronize());
}
