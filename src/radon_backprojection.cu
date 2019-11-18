#include <iostream>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "utils.h"
#include "texture.h"

__global__ void radon_backward_kernel(float *output, cudaTextureObject_t texObj, const float *rays, const float *angles,
                                      const int img_size, const int n_rays, const int n_angles) {

    __shared__ float s_sin[256];
    __shared__ float s_cos[256];

    // Calculate image coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch_id = blockIdx.z;
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    if(tid < n_angles){
        s_sin[tid] = __sinf(angles[tid]);
        s_cos[tid] = __cosf(angles[tid]);
    }
    __syncthreads();

    const float v = img_size / 2;
    const float dx = (float) x - v + 0.5;
    const float dy = (float) y - v + 0.5;

    float tmp = 0.0;
    const float r = hypot(dx, dy);

    if(r <= v){
        for (int i = 0; i < n_angles; i++) {
            float j = s_cos[i] * dx + s_sin[i] * dy + v;
            tmp += tex2DLayered<float>(texObj, j, i + 0.5f, batch_id);
        }
    }

    output[batch_id * img_size * img_size + y * img_size + x] = tmp;
}

void radon_backward_cuda(const float *x, const float *rays, const float *angles, float *y, TextureCache& tex_cache, const int batch_size, const int img_size, const int n_rays, const int n_angles) {
    // copy x into CUDA Array (allocating it if needed) and bind to texture
    tex_cache.put(x, batch_size, n_rays, n_angles, n_rays);

    // Invoke kernel
    const int grid_size = img_size / 16;
    dim3 dimGrid(grid_size, grid_size, batch_size);
    dim3 dimBlock(16, 16);

    radon_backward_kernel <<< dimGrid, dimBlock >>> (y, tex_cache.texObj, rays, angles, img_size, n_rays, n_angles);
}

/*
__global__ void apply_filter(cufftComplex *sino, const int fft_size) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < fft_size) {
        sino[fft_size * y + x].x *= float(x);
        sino[fft_size * y + x].y *= float(x);
    }
}


unsigned int next_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void radon_filter_sinogram_cuda(const float *x, float *y, const int batch_size, const int n_rays, const int n_angles) {
    const int rows = batch_size * n_angles;
    const int padded_size = next_power_of_two(n_rays * 2);
    // cuFFT only stores half of the coefficient because they are symmetric (see cuFFT documentation)
    const int fft_size = padded_size / 2 + 1;
    std::cout << "rows " << rows << std::endl;
    std::cout << "padded_size " << padded_size << std::endl;
    std::cout << "fft_size " << fft_size << std::endl;

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
    checkCudaErrors(cufftPlan1d(&forward_plan, padded_size, CUFFT_R2C, rows));
    checkCudaErrors(cufftPlan1d(&back_plan, padded_size, CUFFT_C2R, rows));

    // do FFT
    checkCudaErrors(cufftExecR2C(forward_plan, padded_data, complex_data));

    // filter in Fourier domain
    apply_filter << < dim3(fft_size / 16 + 1, rows / 16), dim3(16, 16) >> > (complex_data, fft_size);

    // do iFFT
    checkCudaErrors(cufftExecC2R(back_plan, complex_data, filtered_padded_sino));

    // copy unpadded result in y
    checkCudaErrors(cudaMemcpy2D(y, sizeof(float) * n_rays, filtered_padded_sino, sizeof(float) * padded_size,
                                 sizeof(float) * n_rays, rows, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cufftDestroy(forward_plan));
    checkCudaErrors(cufftDestroy(back_plan));
}
*/
