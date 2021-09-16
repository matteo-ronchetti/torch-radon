#ifndef TORCH_RADON_FLOATCAST_H
#define TORCH_RADON_FLOATCAST_H

#include <cuda_fp16.h>

template <typename T>
__device__ T toType(float);

template<>
inline __device__ float toType(float f)
{
    return f;
};

template<>
inline __device__ __half toType(float f)
{
    return __float2half(f);
};

template<>
inline __device__ unsigned short toType(float f)
{
    return static_cast<unsigned short>(f);
};

#endif // TORCH_RADON_FLOATCAST_H
