#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <utility>
#include "utils.h"

class RadonNoiseGenerator {
    curandState **states = nullptr;
    uint seed;

    void set_seed(const uint seed, int device);

public:
    RadonNoiseGenerator(const uint _seed);

    void set_seed(const uint seed);

    curandState *get(int device);

    void add_noise(float *sinogram, const float signal, const float density_normalization, const bool approximate,
                   const uint width, const uint height, int device);

    void emulate_readings(const float *sinogram, int *readings, const float signal, const float density_normalization,
                          const uint width, const uint height, int device);

    void free();

    ~RadonNoiseGenerator();
};

void readings_lookup_cuda(const int *x, float *y, const float *lookup_table, const uint lookup_size, const uint width,
                          const uint height, int device);

std::pair<float, float>
compute_ab(const float *x, const int size, const float signal, const float eps, const int k, const int device);

void compute_ev_lookup(const float *x, const float *weights, float *y_mean, float *y_var, const int size, const int weights_size,
                       const float signal, const int bins, const int k, const int device);