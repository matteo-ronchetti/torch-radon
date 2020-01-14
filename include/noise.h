#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "utils.h"

class RadonNoiseGenerator {
    curandState *states = nullptr;

public:
    RadonNoiseGenerator(const uint seed);

    void set_seed(const uint seed);

    void add_noise(float *sinogram, const float signal, const float density_normalization, const bool approximate,
                   const uint width, const uint height);

    void emulate_readings(const float *sinogram, int *readings, const float signal, const float density_normalization,
                          const uint width, const uint height);

    void free();
};

void readings_lookup_cuda(const int* x, float*  y,const float* lookup_table, const uint lookup_size, const uint width, const uint height);