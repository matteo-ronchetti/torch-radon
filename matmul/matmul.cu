#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

using namespace std;

#define WIDTH 4                      // The vector-width (in number of floats)

//#define TSM 128                      // The tile-size in dimension M
//#define TSN 128                      // The tile-size in dimension N
//#define TSK 16                       // The tile-size in dimension K
//#define WPTM 8                       // The amount of work-per-thread in dimension M
//#define WPTN 8                       // The amount of work-per-thread in dimension N
//#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)
//#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)
//#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
//#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif

//#define MOD2(x, y) ((x) % (y))
//#define DIV2(x, y) ((x) / (y))


__global__ void simple_kernel(const float *X, const float *W, float *Y, const int N, const int channels) {
    const int b = blockIdx.z / channels;
    const int c = blockIdx.z % channels;
    const int c_offset = c * N * N;
    const int bc_offset = (b * channels * N * N + c * N * N);

    const int x = blockIdx.x * 16 + threadIdx.x;
    const int y = blockIdx.y * 16 + threadIdx.y;

    __shared__ float L[16][16];
    __shared__ float R[16][16];

    float tmp = 0.0f;
    for (int tile = 0; tile < N / 16; tile++) {
        L[threadIdx.y][threadIdx.x] = X[bc_offset + y * N + (tile * 16 + threadIdx.x)];
        R[threadIdx.y][threadIdx.x] = W[c_offset + x * N + (tile * 16 + threadIdx.y)];

        __syncthreads();
        for (int k = 0; k < 16; k++) {
            tmp += L[threadIdx.y][k] * R[k][threadIdx.x];
        }
        __syncthreads();
    }

    Y[bc_offset + y * N + x] = tmp;
}

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

void filter_sinogram(const float* sinogram, const float* filters, float* res, const int batch_size, const int channels, const int img_size){
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

int main() {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

    const int batch = 16;
    const int channels = 8;
    const int N = 256;
    float *A, *B, *C;

    cudaMalloc((void **) &A, batch * channels * N * N * sizeof(float));
    cudaMalloc((void **) &B, batch * channels * N * N * sizeof(float));
    cudaMalloc((void **) &C, batch * channels * N * N * sizeof(float));

    float *A_cpu = (float *) malloc(N * N * sizeof(float));
    float *B_cpu = (float *) malloc(N * N * sizeof(float));
    float *C_cpu = (float *) malloc(N * N * sizeof(float));
    float *D_cpu = (float *) malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A_cpu[i] = (float) rand() / (float) RAND_MAX - 0.5;
        B_cpu[i] = (float) rand() / (float) RAND_MAX - 0.5;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += A_cpu[i * N + k] * B_cpu[j * N + k];
            }
            C_cpu[i * N + j] = tmp;
        }
    }

    cudaMemcpy((void *) A, (void *) A_cpu, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) B, (void *) B_cpu, N * N * sizeof(float), cudaMemcpyHostToDevice);

    //simple_kernel << < dim3(N / 16, N / 16, batch * channels), dim3(16, 16) >> > (A, B, C, N, channels);

    filter_sinogram(A, B, C, batch, channels, N);

    cudaMemcpy((void *) D_cpu, (void *) C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    float err = 0;
    for (int i = 0; i < N * N; i++) {
        err += pow(C_cpu[i] - D_cpu[i], 2);
    }

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) cout << C_cpu[(i) * N + (249 + j)] << " ";
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) cout << D_cpu[(i) * N + (249 + j)] << " ";
        cout << endl;
    }
    cout << endl;

    cout << "Error: " << err << endl;

}