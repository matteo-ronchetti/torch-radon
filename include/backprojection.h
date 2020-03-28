#include "texture.h"

void radon_backward_cuda(
        const float *x, const float *rays, const float *angles,
        float *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_rays, const int n_angles, const int device, const bool extend = false
);

//void radon_backward_cuda_lb(
//        const float *x, const float *rays, const float *angles,
//        float *y, TextureCache &tex_cache, const int batch_size,
//        const int img_size, const int n_rays, const int n_angles, const int device, const bool extend = false
//);

// half precision version
void radon_backward_cuda(
        const unsigned short *x, const float *rays, const float *angles,
        unsigned short *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_rays, const int n_angles, const int device, const bool extend = false
);