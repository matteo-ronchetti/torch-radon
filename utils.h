#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

cudaTextureObject_t create_texture(float* data, cudaArray*& cuArray, uint width, uint height, uint pitch){
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, data, pitch*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}
