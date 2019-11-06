#include <torch/extension.h>

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
	//cudaBindTexture2D(0, gT_FanVolumeTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);
	cudaBindTextureToArray(gT_FanVolumeTexture, dataArray, channelDesc);

	// TODO: error value?

	return my_tex;
}

__global__ void transformKernel(float* output, texture<float, 2> texObj) {
    // Calculate texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    output[y * 128 + x] = tex2D(texObj, x+0.5, y+0.5);
}

torch::Tensor copy_image_cuda(torch::Tensor x){
    cudaArray* tmp;
    auto my_tex = bindVolumeDataTexture(x.data<float>(), tmp, 128, 128, 128);

    auto y = torch::zeros_like(x);

    // Invoke kernel
    dim3 dimGrid(8, 8);
    dim3 dimBlock(16, 16);

    transformKernel<<<dimGrid, dimBlock>>>(y.data<float>(), my_tex);

    return y;
}
