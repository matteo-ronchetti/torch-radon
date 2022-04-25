#pragma once
#include <vector>
#include "texture.h"

template<typename T>
void radon_forward_cuda(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const VolumeCfg& vol_cfg, const Projection2D& proj_cfg, const ExecCfg& exec_cfg,
        const int batch_size, const int device
);

template<typename T>
void radon_forward_cuda_3d(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const VolumeCfg& vol_cfg, Projection3D& proj_cfg, const ExecCfg& exec_cfg,
        const int batch_size, const int device
);

void radon_forward_cuda_3d_batch(
        const float *x, const float *angles, float *y, TextureCache &tex_cache,
        const VolumeCfg& vol_cfg, std::vector<Projection3D>& proj_cfg, const ExecCfg& exec_cfg,
        const int batch_size, const int device
);


