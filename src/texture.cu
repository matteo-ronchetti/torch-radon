#include "texture.h"
#include "utils.h"
#include <iostream>
#include <cuda_fp16.h>


cudaChannelFormatDesc get_channel_desc(int channels, int precision) {
    if (precision == PRECISION_FLOAT) {
        if (channels == 1) {
            return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        }
        if (channels == 4) {
            return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        }
    }
    if (precision == PRECISION_HALF && channels == 4) {
        return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    }
    std::cerr << "[TORCH RADON] ERROR unsupported number of channels and precision (channels:" << channels
              << ", precision: " << precision << ")" << std::endl;
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
}

Texture::Texture(DeviceSizeKey k) : key(k) {
    checkCudaErrors(cudaSetDevice(this->key.device));

#ifdef VERBOSE
    std::cout << "[TORCH RADON] Allocating Texture " << this->key << std::endl;
#endif

    // Allocate a layered CUDA array
    cudaChannelFormatDesc channelDesc = get_channel_desc(key.channels, key.precision);
    const cudaExtent extent = make_cudaExtent(k.width, k.height, k.batch / key.channels);
//    std::cout << k << std::endl;
    checkCudaErrors(cudaMalloc3DArray(&array, &channelDesc, extent, cudaArrayLayered));

    // Create resource descriptor
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
    checkCudaErrors(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));

    // Create surface object
    checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
}

__global__ void
write_to_surface(const float *data, cudaSurfaceObject_t surface, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = 4 * blockIdx.z;

    if (x < width && y < height) {
        const int wh = width * height;
        const int offset = b * wh + y * width + x;

        float4 tmp;
        tmp.x = data[0 * wh + offset];
        tmp.y = data[1 * wh + offset];
        tmp.z = data[2 * wh + offset];
        tmp.w = data[3 * wh + offset];

        surf2DLayeredwrite<float4>(tmp, surface, x * sizeof(float4), y, blockIdx.z);
    }
}

void Texture::put(const float *data) {
    if (this->key.precision == PRECISION_HALF) {
        std::cerr << "[TORCH RADON] ERROR putting half precision data into a float texture" << std::endl;
    }

    checkCudaErrors(cudaSetDevice(this->key.device));

    if (key.channels == 1) {
        // copy data into array
        cudaMemcpy3DParms myparms = {0};
        myparms.srcPos = make_cudaPos(0, 0, 0);
        myparms.dstPos = make_cudaPos(0, 0, 0);
        myparms.srcPtr = make_cudaPitchedPtr((void *) data, key.width * sizeof(float), this->key.width,
                                             this->key.height);
        myparms.dstArray = this->array;
        myparms.extent = make_cudaExtent(this->key.width, this->key.height, this->key.batch);
        myparms.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&myparms));
    } else {
        dim3 grid_dim(roundup_div(key.width, 16), roundup_div(key.height, 16), key.batch / 4);
        write_to_surface << < grid_dim, dim3(16, 16) >> > (data, this->surface, key.width, key.height);
    }

}

__global__ void
write_half_to_surface(const __half *data, cudaSurfaceObject_t surface, const int width, const int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = 4 * blockIdx.z;

    if (x < width && y < height) {
        const int wh = width * height;
        const int offset = b * wh + y * width + x;
        __half tmp[4];
        for (int i = 0; i < 4; i++) tmp[i] = __float2half(data[i * wh + offset]);

        surf2DLayeredwrite<float2>(*(float2 *) tmp, surface, x * sizeof(float2), y, blockIdx.z);
    }
}

void Texture::put(const unsigned short *data) {
    if (this->key.precision == PRECISION_FLOAT) {
        std::cerr << "[TORCH RADON] ERROR putting single precision data into a half precision texture" << std::endl;
    }

    checkCudaErrors(cudaSetDevice(this->key.device));

    dim3 grid_dim(roundup_div(key.width, 16), roundup_div(key.height, 16), key.batch / 4);
    write_half_to_surface << < grid_dim, dim3(16, 16) >> > ((__half *) data, this->surface, key.width, key.height);
}

bool Texture::matches(DeviceSizeKey &k) {
    return k == this->key;
}

Texture::~Texture() {
#ifdef VERBOSE
    std::cout << "[TORCH RADON] Freeing Texture " << this->key << std::endl;
#endif
    if (this->array != nullptr) {
        checkCudaErrors(cudaSetDevice(this->key.device));
        checkCudaErrors(cudaDestroyTextureObject(this->texture));
        checkCudaErrors(cudaDestroySurfaceObject(this->surface));
        checkCudaErrors(cudaFreeArray(this->array));
        this->array = nullptr;
    }
}