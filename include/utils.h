#ifndef TORCH_RADON_UTILS_H
#define TORCH_RADON_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


typedef unsigned int uint;

template<typename T> void check_cuda(T result, const char* func, const char* file, const int line){
    if (result){
        fprintf(stderr, "CUDA error at %s (%s:%d) error code: %d, error string: %s\n",
                func, file, line, static_cast<unsigned int>(result), cudaGetErrorString(result));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val)  check_cuda((val), #val, __FILE__, __LINE__)

/*
cudaTextureObject_t
create_texture(const float *data, cudaArray *&cuArray, uint batch_size, uint width, uint height, uint pitch) {
    // Allocate a layered CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const cudaExtent extent = make_cudaExtent(width, height, batch_size);
    checkCudaErrors(cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayLayered));

    // copy data into array
    cudaMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0, 0, 0);
    myparms.dstPos = make_cudaPos(0, 0, 0);
    myparms.srcPtr = make_cudaPitchedPtr((void *) data, pitch * sizeof(float), width, height);
    myparms.dstArray = cuArray;
    myparms.extent = extent;
    myparms.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&myparms));
//    cudaMemcpy2DToArray(cuArray, 0, 0, data, pitch*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice);

    // Specify texture
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    //resDesc.array = cuArray;

    // Specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    return texObj;
}*/

#endif
