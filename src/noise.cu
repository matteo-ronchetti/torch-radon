#include "noise.h"
#include <iostream>

__global__ void initialize_random_states(curandState *state, const uint seed) {
    const uint sequence_id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, sequence_id, 0, &state[sequence_id]);
}

template<bool approximate>
__global__ void
radon_sinogram_noise(float *sinogram, curandState *state, const float signal, const float density_normalization,
                     const uint width, const uint height) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = y * blockDim.x * gridDim.x + x;
    const uint y_step = blockDim.y * gridDim.y;

    if (tid < 128 * 1024) {
        // load curand state in local memory
        curandState localState = state[tid];

        // loop down the sinogram adding noise
        for (uint yy = y; yy < height; yy += y_step) {
            if (x < width) {
                uint pos = yy * width + x;
                // measured signal = signal * exp(-sinogram[pos])
                // then apply poisson noise
                float mu = __expf(signal - sinogram[pos] / density_normalization);
                float reading;
                if (approximate) {
                    float var = __fsqrt_rn(mu);
                    reading = fmaxf(curand_normal(&localState) * var + mu, 1.0f);
                } else {
                    reading = fmaxf(curand_poisson(&localState, mu), 1.0f);
                }

                // convert back to sinogram scale
                sinogram[pos] = fmaxf((signal - __logf(reading)), 0.0f) * density_normalization;
            }
        }

        // save curand state back in global memory
        state[tid] = localState;
    }
}

__global__ void radon_emulate_readings(const float *sinogram, int *readings, curandState *state, const float signal,
                                       const float density_normalization, const uint width, const uint height) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = y * blockDim.x * gridDim.x + x;
    const uint y_step = blockDim.y * gridDim.y;

    // load curand state in local memory
    curandState localState = state[tid];

    // loop down the sinogram adding noise
    for (uint yy = y; yy < height; yy += y_step) {
        uint pos = yy * width + x;
        // measured signal = signal * exp(-sinogram[pos])
        // then apply poisson noise
        float mu = __expf(signal - sinogram[pos] / density_normalization);
        readings[pos] = curand_poisson(&localState, mu);
    }

    // save curand state back in global memory
    state[tid] = localState;
}

RadonNoiseGenerator::RadonNoiseGenerator(const uint _seed) : seed(_seed) {
    this->states = (curandState **) malloc(sizeof(curandState * ) * 8);
    for (int i = 0; i < 8; i++) this->states[i] = nullptr;
}


void RadonNoiseGenerator::set_seed(const uint seed, int device) {
    initialize_random_states << < 128, 1024 >> > (this->get(device), seed);
}

void RadonNoiseGenerator::set_seed(const uint seed) {
    this->seed = seed;
    for (int i = 0; i < 8; i++) {
        if (this->states[i] != nullptr) {
            this->set_seed(seed, i);
        }
    }
}

curandState *RadonNoiseGenerator::get(int device) {
    if (this->states[device] == nullptr) {
        checkCudaErrors(cudaSetDevice(device));
#ifdef VERBOSE
        std::cout << "[TORCH RADON] Allocating Random states on device " << device << std::endl;
#endif

        // allocate random states
        checkCudaErrors(cudaMalloc((void **) &states[device], 128 * 1024 * sizeof(curandState)));
        this->set_seed(seed, device);
    }
    return this->states[device];
}

void RadonNoiseGenerator::add_noise(float *sinogram, const float signal, const float density_normalization,
                                    const bool approximate, const uint width, const uint height, int device) {
    checkCudaErrors(cudaSetDevice(device));

    if (approximate) {
        radon_sinogram_noise<true> << < dim3(width / 16, 8 * 1024 / width), dim3(16, 16) >> >
                                                                            (sinogram, this->get(
                                                                                    device), signal, density_normalization, width, height);
    } else {
        radon_sinogram_noise<false> << < dim3(width / 16, 8 * 1024 / width), dim3(16, 16) >> >
                                                                             (sinogram, this->get(
                                                                                     device), signal, density_normalization, width, height);
    }
}

void RadonNoiseGenerator::emulate_readings(const float *sinogram, int *readings, const float signal,
                                           const float density_normalization, const uint width, const uint height,
                                           int device) {
    checkCudaErrors(cudaSetDevice(device));

    radon_emulate_readings << < dim3(width / 16, 8 * 1024 / width), dim3(16, 16) >> >
                                                                    (sinogram, readings, this->get(
                                                                            device), signal, density_normalization, width, height);
}

void RadonNoiseGenerator::free() {
    for (int i = 0; i < 8; i++) {
        if (this->states[i] != nullptr) {
#ifdef VERBOSE
            std::cout << "[TORCH RADON] Freeing Random states on device " << i << std::endl;
#endif
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cudaFree(this->states[i]));
            this->states[i] = nullptr;
        }
    }
}

RadonNoiseGenerator::~RadonNoiseGenerator() {
    this->free();
}

__global__ void
lookup_kernel(const int *readings, float *result, const float *lookup_table, const uint lookup_size, const uint width,
              const uint height) {
    // TODO use shared memory
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint y_step = blockDim.y * gridDim.y;

    for (uint yy = y; yy < height; yy += y_step) {
        uint pos = yy * width + x;
        int index = min(readings[pos], lookup_size - 1);
        result[pos] = lookup_table[index];
    }
}

void readings_lookup_cuda(const int *x, float *y, const float *lookup_table, const uint lookup_size, const uint width,
                          const uint height, int device) {
    checkCudaErrors(cudaSetDevice(device));

    lookup_kernel << < dim3(width / 16, 8 * 1024 / width), dim3(16, 16) >> >
                                                           (x, y, lookup_table, lookup_size, width, height);
}

__inline__ __device__ void warpReduce(volatile float *sdata, const int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template<int n_threads>
__inline__ __device__ void dualWarpReduce(float *sa, float *sb, const int tid) {
    if (n_threads >= 512) {
        if (tid < 256) {
            sa[tid] += sa[tid + 256];
            sb[tid] += sb[tid + 256];
        }
        __syncthreads();
    }
    if (n_threads >= 256) {
        if (tid < 128) {
            sa[tid] += sa[tid + 128];
            sb[tid] += sb[tid + 128];
        }
        __syncthreads();
    }
    if (n_threads >= 128) {
        if (tid < 64) {
            sa[tid] += sa[tid + 64];
            sb[tid] += sb[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sa, tid);
        warpReduce(sb, tid);
    }
}

__global__ void
ab_kernel(const float *x, const int size, float *ab, const float signal, const float eps, const int k,
          const float nlog) {
    __shared__ float sa[512];
    __shared__ float sb[512];
    const int tid = threadIdx.x;

    float a = 0.0f;
    float b = 0.0f;
    for (int i = tid; i < size; i += 512) {
        float y = x[i];
        float v = exp(float(k * (signal - y) - nlog - exp(signal - y)));
        if (y <= eps) {
            a += v;
        } else {
            b += v;
        }
    }

    sa[tid] = a;
    sb[tid] = b;
    __syncthreads();

    dualWarpReduce<512>(sa, sb, tid);

    if (tid == 0) {
        ab[0] = sa[0];
        ab[1] = sb[0];
    }
}

std::pair<float, float> compute_ab(const float *x, const int size, const float signal, const float eps, const int k,
                                   const int device) {
    checkCudaErrors(cudaSetDevice(device));

    float ab_cpu[2];
    float *ab;
    checkCudaErrors(cudaMalloc(&ab, 2 * sizeof(float)));

    float nlog = k * (signal - eps) - exp(signal - eps);

    ab_kernel << < 1, 512 >> > (x, size, ab, signal, eps, k, nlog);

    checkCudaErrors(cudaMemcpy(ab_cpu, ab, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(ab));

    return std::make_pair(ab_cpu[0], ab_cpu[1]);
}

/*
template<int n_threads>
__global__ void
compute_ev_lookup_kernel(const float *x, const float *norm_p, float *res, const int size, const int norm_p_size,
                         const float signal, const int scale) {
    __shared__ float sp[n_threads];
    __shared__ float spv[n_threads];
    __shared__ float normal_prob[64];

    const int bin = blockIdx.x;
    const int tid = threadIdx.x;

    const float mean_k = bin * scale + (scale - 1.0f) / 2.0f;
    const float normalizer = mean_k > 0 ? mean_k * log(mean_k) - mean_k : 0.0f;

    // load normal weights into shared memory
    if (tid < norm_p_size) {
        normal_prob[tid] = norm_p[tid];
    }
    __syncthreads();


    float p = 0.0f;
    float pv = 0.0f;
    for (int i = tid; i < size; i += n_threads) {
        // read sinogram value and precompute
        const float y = x[i];
        const float delta = signal - y;
        const float constant_part = bin * scale * delta - exp(delta) - normalizer;

        for (int j = 0; j < norm_p_size; j++) {
            const float pn = normal_prob[j];
            for (int k = 0; k < scale; k++) {
                const float prob = pn * (exp((k + j) * delta + constant_part) + exp((k - j) * delta + constant_part));
                p += prob;
                pv += prob * y;
            }
        }
    }

    sp[tid] = p;
    spv[tid] = pv;
    __syncthreads();

    dualWarpReduce<n_threads>(sp, spv, tid);

    if (tid == 0) {
        res[bin] = spv[0] / sp[0];
    }
}

void compute_ev_lookup(const float *x, const float *norm_p, float *y, const int size, const int norm_p_size,
                       const float signal, const int bins, const int k, const int device) {
    checkCudaErrors(cudaSetDevice(device));

    compute_ev_lookup_kernel<256> << < bins, 256 >> > (x, norm_p, y, size, norm_p_size, signal, k / bins);
}
*/

template<int unroll, bool variance>
__global__ void
compute_ev_lookup_kernel(const float *x, const float *g_weights, const float* mean_estimator, float *res, const int size, const int weights_size,
                         const float signal, const int scale) {
    constexpr int n_threads = 256;
    __shared__ float sp[n_threads];
    __shared__ float spv[n_threads];
    __shared__ float weights[256];

    const int bin = blockIdx.x;
    const int tid = threadIdx.x;

    const float mean_k = bin * scale + (scale - 1.0f) / 2.0f;
    const float normalizer = mean_k > 0 ? mean_k * log(mean_k) - mean_k : 0.0f;
    const int r = (weights_size - scale)/2;

    float estimated_mean;
    if(variance){
        estimated_mean = mean_estimator[bin];
    }

    // load weights into shared memory
    if (tid < weights_size) {
        weights[tid] = g_weights[tid];
    }
    __syncthreads();


    float p = 0.0f;
    float pv = 0.0f;
    for (int i = tid; i < size; i += n_threads) {
        // read sinogram value and precompute
        const float y = x[i];
        const float delta = signal - y;
        const float constant_part = bin * scale * delta - __expf(delta) - normalizer;

        float prob = 0.0f;
        for (int j = 0; j < weights_size; j += unroll) {
            float tmp = (j - r) * delta + constant_part;
#pragma unroll
            for(int h = 0; h < unroll; h++){
                prob += weights[j+h] * __expf(tmp);
                tmp += delta;
            }
        }
        p += prob;
        if(variance) {
            pv += prob * (y - estimated_mean) * ( y - estimated_mean);
        }else{
            pv += prob * y;
        }
    }

    sp[tid] = p;
    spv[tid] = pv;
    __syncthreads();

    dualWarpReduce<n_threads>(sp, spv, tid);

    if (tid == 0) {
        if(variance) {
            res[bin] = sqrt(spv[0] / sp[0]);
        }else{
            res[bin] = spv[0] / sp[0];
        }
    }
}

void compute_ev_lookup(const float *x, const float *weights, float *y_mean, float *y_var, const int size, const int weights_size,
                         const float signal, const int bins, const int k, const int device){
    checkCudaErrors(cudaSetDevice(device));

    if(weights_size % 17 == 0) {
        compute_ev_lookup_kernel<17, false> << < bins, 256 >> > (x, weights, NULL, y_mean, size, weights_size, signal, k / bins);
        compute_ev_lookup_kernel<17, true> << < bins, 256 >> > (x, weights, y_mean, y_var, size, weights_size, signal, k / bins);
        return;
    }

    if(weights_size % 11 == 0) {
        compute_ev_lookup_kernel<11, false> << < bins, 256 >> > (x, weights, NULL, y_mean, size, weights_size, signal, k / bins);
        compute_ev_lookup_kernel<11, true> << < bins, 256 >> > (x, weights, y_mean, y_var, size, weights_size, signal, k / bins);
        return;
    }

    if(weights_size % 7 == 0) {
        compute_ev_lookup_kernel<7, false> << < bins, 256 >> > (x, weights, NULL, y_mean, size, weights_size, signal, k / bins);
        compute_ev_lookup_kernel<7, true> << < bins, 256 >> > (x, weights, y_mean, y_var, size, weights_size, signal, k / bins);
        return;
    }

    if(weights_size % 4 == 0) {
        compute_ev_lookup_kernel<4, false> << < bins, 256 >> > (x, weights, NULL, y_mean, size, weights_size, signal, k / bins);
        compute_ev_lookup_kernel<4, true> << < bins, 256 >> > (x, weights, y_mean, y_var, size, weights_size, signal, k / bins);
        return;
    }

    if(weights_size % 2 == 0) {
        compute_ev_lookup_kernel<2, false> << < bins, 256 >> > (x, weights, NULL, y_mean, size, weights_size, signal, k / bins);
        compute_ev_lookup_kernel<2, true> << < bins, 256 >> > (x, weights, y_mean, y_var, size, weights_size, signal, k / bins);
        return;
    }

    compute_ev_lookup_kernel<1, false> << < bins, 256 >> > (x, weights, NULL, y_mean, size, weights_size, signal, k / bins);
    compute_ev_lookup_kernel<1, true> << < bins, 256 >> > (x, weights, y_mean, y_var, size, weights_size, signal, k / bins);
}
