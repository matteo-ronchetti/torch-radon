#ifndef TORCH_RADON_TEXTURE_CACHE_H
#define TORCH_RADON_TEXTURE_CACHE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "cache.h"

#define PRECISION_FLOAT 1
#define PRECISION_HALF 0

class Texture {
    cudaArray *array = nullptr;
    DeviceSizeKey key;

public:
    cudaSurfaceObject_t surface = 0;
    cudaTextureObject_t texture = 0;

    Texture(DeviceSizeKey key);
    void put(const float *data);
    void put(const unsigned short *data);

    bool matches(DeviceSizeKey& k);

    ~Texture();
};


typedef Cache<DeviceSizeKey, Texture> TextureCache;

#endif
