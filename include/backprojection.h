#include "texture.h"

void radon_backward_cuda(
        const float *x, const int det_count, const float det_spacing, const float *angles,
        float *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_angles, const int device, const bool clip_to_circle
);

// half precision version
void radon_backward_cuda(
        const unsigned short *x, const int det_count, const float det_spacing, const float *angles,
        unsigned short *y, TextureCache &tex_cache, const int batch_size,
        const int img_size, const int n_angles, const int device, const bool clip_to_circle
);