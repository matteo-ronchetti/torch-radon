#ifndef TORCH_RADON_UTILS_H
#define TORCH_RADON_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>


typedef unsigned int uint;

class RaysCfg {
public:
    int width;
    int height;

    int det_count;
    float det_spacing;

    int n_angles;
    bool clip_to_circle;

    // source and detector distances (for fanbeam)
    float s_dist = 0.0;
    float d_dist = 0.0;

    bool is_fanbeam = false;

    RaysCfg(int w, int h, int dc, float ds, int na, bool ctc) : width(w), height(h), det_count(dc), det_spacing(ds),
                                                      n_angles(na), clip_to_circle(ctc) {}

    RaysCfg(int w, int h, int dc, float ds, int na, bool ctc, float sd, float dd) : width(w), height(h), det_count(dc),
                                                                          det_spacing(ds),
                                                                          n_angles(na), clip_to_circle(ctc), s_dist(sd),
                                                                          d_dist(dd), is_fanbeam(true) {}
};

inline int roundup_div(const int x, const int y) {
    return x / y + (x % y != 0);
}

inline unsigned int next_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

template<typename T>
void check_cuda(T result, const char *func, const char *file, const int line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s (%s:%d) error code: %d, error string: %s\n",
                func, file, line, static_cast<unsigned int>(result), cudaGetErrorString(result));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val)  check_cuda((val), #val, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        default:
            return "OTHER_ERROR";
    }

    return "<unknown>";
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)

inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
    if (CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n", __FILE__, __LINE__, err,
                _cudaGetErrorEnum(err));
        cudaDeviceReset();
    }
}

#endif
