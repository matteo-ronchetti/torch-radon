#include "radon_noise.h"
#include <curand_kernel.h>


__global__ void initialize_random_states(curandState *state, const uint seed){
    const uint sequence_id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, sequence_id, 0, &state[sequence_id]);
}

__global__ void radon_sinogram_noise(float* sinogram, curandState *state, const float sino_max, const float signal, const uint width, const uint height){
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint tid = y * blockDim.x * gridDim.x + x;
    const uint y_step = blockDim.y * gridDim.y;

    // load curand state in local memory
    curandState localState = state[tid];

    // loop through down the sinogram adding noise
    for(uint yy = y; yy < height; yy += y_step){
        uint pos = yy * width + x;
        float reading = curand_poisson(&localState, signal * exp(-sinogram[pos]/sino_max));
        sinogram[pos] = -sino_max * log(reading / signal);
    }

    // save curand state back in global memory
    state[tid] = localState;
}
