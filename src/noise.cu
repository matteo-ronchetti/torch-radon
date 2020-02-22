#include "noise.h"
#include <iostream>

__global__ void initialize_random_states(curandState *state, const uint seed){
    const uint sequence_id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, sequence_id, 0, &state[sequence_id]);
}

template<bool approximate> __global__ void radon_sinogram_noise(float* sinogram, curandState *state, const float signal, const float density_normalization, const uint width, const uint height){
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = y * blockDim.x * gridDim.x + x;
    const uint y_step = blockDim.y * gridDim.y;

    // load curand state in local memory
    curandState localState = state[tid];

    // loop down the sinogram adding noise
    for(uint yy = y; yy < height; yy += y_step){
        uint pos = yy * width + x;
        // measured signal = signal * exp(-sinogram[pos])
        // then apply poisson noise
        float mu = __expf(signal - sinogram[pos]/density_normalization);
        float reading;
        if(approximate){
            float var = __fsqrt_rn(mu);
            reading = fmaxf(curand_normal(&localState)*var + mu, 1.0f);
        }else{
            reading = fmaxf(curand_poisson(&localState, mu), 1.0f);
        }

        // convert back to sinogram scale
        sinogram[pos] = fmaxf((signal -__logf(reading)), 0.0f) * density_normalization;
    }

    // save curand state back in global memory
    state[tid] = localState;
}

__global__ void radon_emulate_readings(const float* sinogram, int* readings, curandState *state, const float signal, const float density_normalization, const uint width, const uint height){
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = y * blockDim.x * gridDim.x + x;
    const uint y_step = blockDim.y * gridDim.y;

    // load curand state in local memory
    curandState localState = state[tid];

    // loop down the sinogram adding noise
    for(uint yy = y; yy < height; yy += y_step){
        uint pos = yy * width + x;
        // measured signal = signal * exp(-sinogram[pos])
        // then apply poisson noise
        float mu = __expf(signal - sinogram[pos]/density_normalization);
        readings[pos] = curand_poisson(&localState, mu);
    }

    // save curand state back in global memory
    state[tid] = localState;
}

RadonNoiseGenerator::RadonNoiseGenerator(const uint seed){
    // TODO
    checkCudaErrors(cudaSetDevice(0));

    // allocate random states
    checkCudaErrors(cudaMalloc((void **)&states, 128*1024 * sizeof(curandState)));

    this->set_seed(seed);
}

void RadonNoiseGenerator::set_seed(const uint seed){
    // TODO
    checkCudaErrors(cudaSetDevice(0));

    initialize_random_states<<<128,1024>>>(states, seed);
}

void RadonNoiseGenerator::add_noise(float* sinogram, const float signal, const float density_normalization, const bool approximate, const uint width, const uint height){
    // TODO
    checkCudaErrors(cudaSetDevice(0));

    if(approximate){
        radon_sinogram_noise<true><<<dim3(width/64, 32*1024/width), dim3(64, 4)>>>(sinogram, states, signal, density_normalization, width, height);
    }else{
        radon_sinogram_noise<false><<<dim3(width/64, 32*1024/width), dim3(64, 4)>>>(sinogram, states, signal, density_normalization, width, height);
    }
}

void RadonNoiseGenerator::emulate_readings(const float* sinogram, int* readings, const float signal, const float density_normalization, const uint width, const uint height){
    // TODO
    checkCudaErrors(cudaSetDevice(0));

    radon_emulate_readings<<<dim3(width/64, 32*1024/width), dim3(64, 4)>>>(sinogram, readings, states, signal, density_normalization, width, height);
}

void RadonNoiseGenerator::free(){
    if(this->states != nullptr){
        checkCudaErrors(cudaFree(this->states));
    }
}

__global__ void lookup_kernel(const int* readings, float *result, const float* lookup_table, const uint lookup_size, const uint width, const uint height){
    // TODO use shared memory
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint y_step = blockDim.y * gridDim.y;

    for(uint yy = y; yy < height; yy += y_step){
        uint pos = yy * width + x;
        int index = min(readings[pos], lookup_size-1);
        result[pos] = lookup_table[index];
    }
}

void readings_lookup_cuda(const int* x, float*  y,const float* lookup_table, const uint lookup_size, const uint width, const uint height){
    // TODO
    checkCudaErrors(cudaSetDevice(0));

    lookup_kernel<<<dim3(width/64, 32*1024/width), dim3(64, 4)>>>(x, y, lookup_table, lookup_size, width, height);
}
