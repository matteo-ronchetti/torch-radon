#ifndef TORCH_RADON_FILTERING_H
#define TORCH_RADON_FILTERING_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "cache.h"


class FFTStructures {
public:
    cufftReal *padded_data = nullptr;
    cufftComplex *complex_data = nullptr;
    cufftReal *filtered_padded_sino = nullptr;
    cufftHandle forward_plan, back_plan;

    DeviceSizeKey key;

    int n_angles; // = key.height;
    int n_rays; // = key.width;
    int rows; // = key.batch * n_angles;
    int padded_size; // = next_power_of_two(n_rays * 2);
    int fft_size;
    FFTStructures(DeviceSizeKey key);

    void FFT(const float *x);

    void iFFT(float *y);

    bool matches(DeviceSizeKey &k);

    ~FFTStructures();
};


typedef Cache<DeviceSizeKey, FFTStructures> FFTCache;

void radon_filter_sinogram_cuda(const float *x, float *y, FFTCache &fft_cache, const int batch_size, const int n_angles,
                                const int n_rays, const int device);

#endif //TORCH_RADON_FILTERING_H
