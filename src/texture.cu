#include "texture.h"
#include <iostream>

SingleTextureCache::SingleTextureCache(){}

void SingleTextureCache::free(){
    //std::cout << "Free" << std::endl;
    if(this->array != nullptr){
        checkCudaErrors(cudaFreeArray(this->array));
        checkCudaErrors(cudaDestroyTextureObject(this->texObj));
        this->array = nullptr;
    }
}

void SingleTextureCache::allocate(uint b, uint w, uint h){
    // free previously allocated array
    this->free();

    this->batch_size = b;
    this->width = w;
    this->height = h;

    // Allocate a layered CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const cudaExtent extent = make_cudaExtent(width, height, batch_size);
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

void SingleTextureCache::put(const float *data, uint b, uint w, uint h, uint pitch){
    // only reallocate when required
    if(this->batch_size != b || this->width != w ||  this->height != h){
        std::cout << "Alloc" << std::endl;
        std::cout << this->batch_size << " " << b << " " <<  this->width  << " " <<  w  << " " <<  this->height  << " " <<  h << std::endl;
        this->allocate(b, w, h);
    }

    // copy data into array
    cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0, 0, 0);
    myparms.dstPos = make_cudaPos(0, 0, 0);
    myparms.srcPtr = make_cudaPitchedPtr((void *) data, pitch * sizeof(float), width, height);
    myparms.dstArray = this->array;
    myparms.extent = make_cudaExtent(width, height, batch_size);
    myparms.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&myparms));
}



TextureCache::TextureCache(){
    this->caches = (SingleTextureCache**) malloc(8*sizeof(SingleTextureCache*));
    memset(this->caches, 0, 8*sizeof(SingleTextureCache*));
};

void TextureCache::put(const float *data, uint b, uint w, uint h, uint pitch, int device){
    if(this->caches[device] == 0){
        std::cout << "NEW TEXTURE CACHE " << device << std::endl;
        this->caches[device] = new SingleTextureCache();
    }
    this->caches[device]->put(data, b, w, h, pitch);
}

void TextureCache::free(){
    for(int device = 0; device < 8; device++){
        if(this->caches[device] != 0){
            checkCudaErrors(cudaSetDevice(device));
            this->caches[device]->free();
        }
    }
}
    
cudaTextureObject_t TextureCache::texObj(int device){
    return this->caches[device]->texObj;
}