#include "texture.h"
#include <iostream>

Texture::Texture(DeviceSizeKey k) : key(k) {
    checkCudaErrors(cudaSetDevice(this->key.device));

#ifdef VERBOSE
    std::cout << "[TORCH RADON] Allocating Texture " << this->key << std::endl;
#endif

    // Allocate a layered CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const cudaExtent extent = make_cudaExtent(k.width, k.height, k.batch);
    checkCudaErrors(cudaMalloc3DArray(&array, &channelDesc, extent, cudaArrayLayered));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
}

void Texture::put(const float *data) {
    checkCudaErrors(cudaSetDevice(this->key.device));
    const uint pitch = this->key.width;

    // copy data into array
    cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0, 0, 0);
    myparms.dstPos = make_cudaPos(0, 0, 0);
    myparms.srcPtr = make_cudaPitchedPtr((void *) data, pitch * sizeof(float), this->key.width, this->key.height);
    myparms.dstArray = this->array;
    myparms.extent = make_cudaExtent(this->key.width, this->key.height, this->key.batch);
    myparms.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&myparms));
}

bool Texture::matches(DeviceSizeKey& k){
    return k == this->key;
}

Texture::~Texture() {
#ifdef VERBOSE
    std::cout << "[TORCH RADON] Freeing Texture " << this->key << std::endl;
#endif
    if (this->array != nullptr) {
        checkCudaErrors(cudaSetDevice(this->key.device));
        checkCudaErrors(cudaFreeArray(this->array));
        checkCudaErrors(cudaDestroyTextureObject(this->texObj));
        this->array = nullptr;
    }
}