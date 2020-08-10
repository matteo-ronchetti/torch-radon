#include "texture.h"

template< typename T> void radon_backward_cuda(
        const T *x, const float *angles, T *y, TextureCache &tex_cache,
        const RaysCfg& rays_cfg, const int batch_size, const int device
);