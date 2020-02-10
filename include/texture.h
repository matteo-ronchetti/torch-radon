#ifndef TORCH_RADON_TEXTURE_CACHE_H
#define TORCH_RADON_TEXTURE_CACHE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"


class SingleTextureCache{
    cudaArray* array = nullptr;
    uint batch_size = 0;
    uint width = 0;
    uint height = 0;

    public:
    cudaTextureObject_t texObj;

    SingleTextureCache();

    void allocate(uint b, uint w, uint h);

    void put(const float *data, uint b, uint w, uint h, uint pitch);

    void free();
};


class TextureCache{
    SingleTextureCache** caches;
    
    public:
    TextureCache();

    void put(const float *data, uint b, uint w, uint h, uint pitch, int device);

    void free();
    
    cudaTextureObject_t texObj(int device);
};

#endif
