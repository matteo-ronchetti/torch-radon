#ifndef TORCH_RADON_TEXTURE_CACHE_H
#define TORCH_RADON_TEXTURE_CACHE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "cache.h"

class Texture {
    cudaArray *array = nullptr;
    DeviceSizeKey key;

public:
    cudaTextureObject_t texObj;

    Texture(DeviceSizeKey key);
    void put(const float *data);

    bool matches(DeviceSizeKey& k);

    ~Texture();
};


typedef Cache<DeviceSizeKey, Texture> TextureCache;

#endif
