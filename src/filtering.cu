#include <iostream>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "utils.h"


__global__ void apply_filter(cufftComplex *sino, const int fft_size, const float scaling) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < fft_size) {
        sino[fft_size * y + x].x *= float(x) / scaling;
        sino[fft_size * y + x].y *= float(x) / scaling;
    }
}


void radon_filter_sinogram_cuda(const float *x, float *y, const int batch_size, const int n_angles, const int n_rays) {
    const int rows = batch_size * n_angles;
    const int padded_size = next_power_of_two(n_rays * 2);
    // cuFFT only stores half of the coefficient because they are symmetric (see cuFFT documentation)
    const int fft_size = padded_size / 2 + 1;

    // pad x
    cufftReal *padded_data = nullptr;
    checkCudaErrors(cudaMalloc((void **) &padded_data, sizeof(cufftReal) * rows * padded_size));
    checkCudaErrors(cudaMemset(padded_data, 0, sizeof(cufftReal) * rows * padded_size));
    checkCudaErrors(cudaMemcpy2D(padded_data, sizeof(cufftReal) * padded_size, x, sizeof(float) * n_rays,
                                 sizeof(float) * n_rays, rows, cudaMemcpyDeviceToDevice));

    // allocate complex tensor to store FFT coefficients
    cufftComplex *complex_data = nullptr;
    checkCudaErrors(cudaMalloc((void **) &complex_data, sizeof(cufftComplex) * rows * fft_size));

    // allocate real tensor to store padded filtered sinogram
    cufftReal *filtered_padded_sino = nullptr;
    checkCudaErrors(cudaMalloc((void **) &filtered_padded_sino, sizeof(cufftReal) * rows * padded_size));
    checkCudaErrors(cudaMemset(filtered_padded_sino, 0, sizeof(cufftReal) * rows * padded_size));

    // create plans for FFT and iFFT
    cufftHandle forward_plan, back_plan;
    cufftSafeCall(cufftPlan1d(&forward_plan, padded_size, CUFFT_R2C, rows));
    cufftSafeCall(cufftPlan1d(&back_plan, padded_size, CUFFT_C2R, rows));

    // do FFT
    cufftSafeCall(cufftExecR2C(forward_plan, padded_data, complex_data));

    // filter in Fourier domain
    apply_filter << < dim3(fft_size / 16 + 1, rows / 16), dim3(16, 16) >> > (complex_data, fft_size, padded_size*padded_size);

    // do iFFT
    cufftSafeCall(cufftExecC2R(back_plan, complex_data, filtered_padded_sino));

    // copy unpadded result in y
    checkCudaErrors(cudaMemcpy2D(y, sizeof(float) * n_rays, filtered_padded_sino, sizeof(float) * padded_size,
                                 sizeof(float) * n_rays, rows, cudaMemcpyDeviceToDevice));

    cufftSafeCall(cufftDestroy(forward_plan));
    cufftSafeCall(cufftDestroy(back_plan));
}