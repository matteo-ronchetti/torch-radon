#ifndef TORCH_RADON_FILTERING_H
#define TORCH_RADON_FILTERING_H

void radon_filter_sinogram_cuda(const float *x, float *y, const int batch_size, const int n_angles, const int n_rays);

#endif //TORCH_RADON_FILTERING_H
