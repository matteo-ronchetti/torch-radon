#include "texture.h"

void radon_backward_cuda(
        const float *x, const float *rays, const float *angles,
        float *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_rays, const int n_angles, const int device
);