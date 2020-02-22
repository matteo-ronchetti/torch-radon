#include <iostream>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "utils.h"
#include "filtering.h"

/*
#define WIDTH 4                      // The vector-width (in number of floats)

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif

template<int TS, int work_per_thread>
__global__ void fast_kernel(const floatX *A, const floatX *B, float *C, const int N, const int channels) {
    // tile size on k (accumulation) dimension
    constexpr int TSK = (128 * 16) / TS;
    // size of the small tile computed by each thread
    constexpr int thread_ts = TS / work_per_thread;

    // blockDim.x = blockDim.y = TS / work_per_thread ==> we have (TS/work_per_thread)**2 threads
    // need to read tiles of size TS*TSK ==> TS*TSK/WIDTH reads
    // Each thread must do (TS*TSK*work_per_thread**2) / (WIDTH * TS * TS) reads (i.e. the following line of code)
    constexpr int loads_per_thread = (TSK * work_per_thread * work_per_thread) / (TS * WIDTH);

    const int b = blockIdx.z / channels;
    const int c = blockIdx.z % channels;
    const int c_offset = c * N * N / WIDTH;
    const int bc_offset = (b * channels * N * N + c * N * N) / WIDTH;


    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int n_tiles = N / TSK;

    // shared memory tiles
    __shared__ float A_shared[TSK][TS + 2];
    __shared__ float B_shared[TSK][TS + 2];

    // registry tiles
    float Areg;
    float Breg[work_per_thread];
    float acc[work_per_thread][work_per_thread];

    for (int wm = 0; wm < work_per_thread; wm++) {
        for (int wn = 0; wn < work_per_thread; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // loop over tiles
    for (int tile = 0; tile < n_tiles; tile++) {
        // Load one tile of A and B into shared memory
        for (int i = 0; i < loads_per_thread; i++) {
            int tile_y = i * TS / loads_per_thread + tid / (TSK / WIDTH);
            int tile_x = tid % (TSK / WIDTH);
            int ay = blockIdx.x * TS + tile_y;
            int by = blockIdx.y * TS + tile_y;
            int kx = tile * TSK / WIDTH + tile_x;

            floatX vecA = __ldg(&A[bc_offset + ay * (N / WIDTH) + kx]);
            floatX vecB = __ldg(&B[c_offset + by * (N / WIDTH) + kx]);

            // Store the loaded vectors into local memory
#if WIDTH == 1
            A_shared[tile_x][tile_y] = vecA;
#elif WIDTH == 2
            A_shared[WIDTH*tile_x + 0][tile_y] = vecA.x;
            A_shared[WIDTH*tile_x + 1][tile_y] = vecA.y;
#elif WIDTH == 4
            A_shared[WIDTH * tile_x + 0][tile_y] = vecA.x;
            A_shared[WIDTH * tile_x + 1][tile_y] = vecA.y;
            A_shared[WIDTH * tile_x + 2][tile_y] = vecA.z;
            A_shared[WIDTH * tile_x + 3][tile_y] = vecA.w;
#endif
#if WIDTH == 1
            B_shared[tile_x][tile_y] = vecB;
#elif WIDTH == 2
            B_shared[WIDTH*tile_x + 0][tile_y] = vecB.x;
            B_shared[WIDTH*tile_x + 1][tile_y] = vecB.y;
#elif WIDTH == 4
            B_shared[WIDTH * tile_x + 0][tile_y] = vecB.x;
            B_shared[WIDTH * tile_x + 1][tile_y] = vecB.y;
            B_shared[WIDTH * tile_x + 2][tile_y] = vecB.z;
            B_shared[WIDTH * tile_x + 3][tile_y] = vecB.w;
#endif
        }

        __syncthreads();

        for (int k = 0; k < TSK; k++) {
            // load a row of B
            for (int j = 0; j < work_per_thread; j++) {
                Breg[j] = B_shared[k][thread_ts * j + threadIdx.y];
            }
            // shuffle doesn't help
            //float val = B_shared[k][thread_ts * (threadIdx.x % work_per_thread) + threadIdx.y];
            //for (int j = 0; j < work_per_thread; j++) {
            //    Breg[j] = __shfl_sync(0xffffffff, val, j, work_per_thread);
            //}

            for (int i = 0; i < work_per_thread; i++) {
                Areg = A_shared[k][thread_ts * i + threadIdx.x];
                for (int j = 0; j < work_per_thread; j++) {
                    acc[i][j] += Areg * Breg[j];
                }
            }
        }

        __syncthreads();
    }

    // save results
    for (int wm = 0; wm < work_per_thread; wm++) {
        int globalRow = blockIdx.x * TS + thread_ts * wm + threadIdx.x;
        for (int wn = 0; wn < work_per_thread; wn++) {
            int globalCol = blockIdx.y * TS + thread_ts * wn + threadIdx.y;
            C[bc_offset * WIDTH + globalRow * N + globalCol] = acc[wm][wn];
        }
    }
}


void sinofilter_cuda(const float* sinogram, const float* filters, float* res, const int batch_size, const int channels, const int img_size){
    // TODO handle not power of two image sizes

    if(img_size == 64){
        fast_kernel<64, 8> <<< dim3(1, 1, batch_size * channels), dim3(16, 16) >>>((const floatX *) sinogram, (const floatX *) filters, res, img_size, channels);
    }else{
        constexpr int tile_size = 64;
        constexpr int work_per_thread = 8;
        dim3 blocks(img_size / tile_size, img_size / tile_size, batch_size * channels);
        dim3 threads(tile_size / work_per_thread, tile_size / work_per_thread);
        fast_kernel<tile_size, work_per_thread> <<<blocks, threads>>>
        ((const floatX *) sinogram, (const floatX *) filters, res, img_size, channels);
    }
}
*/


FFTStructures::FFTStructures(DeviceSizeKey k) : key(k) {
    checkCudaErrors(cudaSetDevice(this->key.device));
#ifdef VERBOSE
    std::cout << "[TORCH RADON] Allocating FFT " << this->key << std::endl;
#endif

    this->n_angles = key.height;
    this->n_rays = key.width;
    this->rows = key.batch * n_angles;
    this->padded_size = next_power_of_two(n_rays * 2);

    // cuFFT only stores half of the coefficient because they are symmetric (see cuFFT documentation)
    this->fft_size = padded_size / 2 + 1;

    // allocate padded version of x
    checkCudaErrors(cudaMalloc((void **) &padded_data, sizeof(cufftReal) * rows * padded_size));
    checkCudaErrors(cudaMemset(padded_data, 0, sizeof(cufftReal) * rows * padded_size));

    // allocate complex tensor to store FFT coefficients
    checkCudaErrors(cudaMalloc((void **) &complex_data, sizeof(cufftComplex) * rows * fft_size));

    // allocate real tensor to store padded filtered sinogram
    checkCudaErrors(cudaMalloc((void **) &filtered_padded_sino, sizeof(cufftReal) * rows * padded_size));
    checkCudaErrors(cudaMemset(filtered_padded_sino, 0, sizeof(cufftReal) * rows * padded_size));

    // create plans for FFT and iFFT
    cufftSafeCall(cufftPlan1d(&forward_plan, padded_size, CUFFT_R2C, rows));
    cufftSafeCall(cufftPlan1d(&back_plan, padded_size, CUFFT_C2R, rows));
}


void FFTStructures::FFT(const float *x) {
    checkCudaErrors(cudaMemcpy2D(padded_data, sizeof(cufftReal) * padded_size, x, sizeof(float) * n_rays,
                                 sizeof(float) * n_rays, rows, cudaMemcpyDeviceToDevice));

    // do FFT
    cufftSafeCall(cufftExecR2C(forward_plan, padded_data, complex_data));
}

void FFTStructures::iFFT(float *y) {
    // do iFFT
    cufftSafeCall(cufftExecC2R(back_plan, complex_data, filtered_padded_sino));

    // copy unpadded result in y
    checkCudaErrors(cudaMemcpy2D(y, sizeof(float) * n_rays, filtered_padded_sino, sizeof(float) * padded_size,
                                 sizeof(float) * n_rays, rows, cudaMemcpyDeviceToDevice));
}

bool FFTStructures::matches(DeviceSizeKey &k) {
    return k == this->key;
}

FFTStructures::~FFTStructures() {
    if (padded_data != nullptr) {
#ifdef VERBOSE
        std::cout << "[TORCH RADON] Freeing FFT " << this->key << std::endl;
#endif
        cudaFree(padded_data);
        cudaFree(complex_data);
        cudaFree(filtered_padded_sino);
        cufftSafeCall(cufftDestroy(forward_plan));
        cufftSafeCall(cufftDestroy(back_plan));
    }
}

__global__ void apply_filter(cufftComplex *sino, const int fft_size, const int rows, const float scaling) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < fft_size && y < rows) {
        sino[fft_size * y + x].x *= float(x) / scaling;
        sino[fft_size * y + x].y *= float(x) / scaling;
    }
}

void
radon_filter_sinogram_cuda(const float *x, float *y, FFTCache &fft_cache, const int batch_size, const int n_angles,
                           const int n_rays, const int device) {

    FFTStructures *fft = fft_cache.get({device, batch_size, n_rays, n_angles});

    checkCudaErrors(cudaSetDevice(device));
    fft->FFT(x);

    // filter in Fourier domain
    const float scaling = fft->padded_size * fft->padded_size;

    dim3 grid(fft->fft_size / 16 + 1, fft->rows / 16 + (fft->rows % 16 != 0));
    apply_filter << < grid, dim3(16, 16) >> > (fft->complex_data, fft->fft_size, fft->rows, scaling);


    fft->iFFT(y);
}