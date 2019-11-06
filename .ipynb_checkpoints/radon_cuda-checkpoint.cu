//#include <torch/extension.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

texture<float, 2> bindVolumeDataTexture(float* data, cudaArray*& dataArray, unsigned int pitch, unsigned int width, unsigned int height)
{
    texture<float, 2> my_tex;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	dataArray = 0;
	cudaMallocArray(&dataArray, &channelDesc, width, height);
	cudaMemcpy2DToArray(dataArray, 0, 0, data, pitch*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice);

	my_tex.addressMode[0] = cudaAddressModeBorder;
	my_tex.addressMode[1] = cudaAddressModeBorder;
	my_tex.filterMode = cudaFilterModeLinear;
	my_tex.normalized = false;

	// TODO: For very small sizes (roughly <=512x128) with few angles (<=180)
	// not using an array is more efficient.
	//cudaBindTexture2D(0, my_tex, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);
	cudaBindTextureToArray(my_tex, dataArray, channelDesc);

	// TODO: error value?

	return my_tex;
}

__global__ void transformKernel(float* output, cudaTextureObject_t texObj) {
    // Calculate texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    output[y * 128 + x] = tex2D<float>(texObj, x+0.5, y+0.5);
}

cudaTextureObject_t create_texture(float* data, unsigned int width, unsigned int height){
    unsigned int pitch = width;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);
    cudaArray* cuArray;
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

//torch::Tensor copy_image_cuda(torch::Tensor x){
void copy_image_cuda(float* x, float* y){
    //cudaArray* tmp;
    //auto my_tex = bindVolumeDataTexture(x.data<float>(), tmp, 128, 128, 128);

    //auto y = torch::zeros_like(x);
    
    auto my_tex = create_texture(x, 128, 128);

    // Invoke kernel
    dim3 dimGrid(8, 8);
    dim3 dimBlock(16, 16);

    transformKernel<<<dimGrid, dimBlock>>>(y, my_tex);

    //return y;
}

int main(){
    std::cout << "Hello CUDA" << std::endl;
}