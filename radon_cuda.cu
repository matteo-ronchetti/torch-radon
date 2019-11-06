#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

static texture<float, 2> gT_FanVolumeTexture;


static bool bindVolumeDataTexture(float* data, cudaArray*& dataArray, unsigned int pitch, unsigned int width, unsigned int height)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	dataArray = 0;
	cudaMallocArray(&dataArray, &channelDesc, width, height);
	cudaMemcpy2DToArray(dataArray, 0, 0, data, pitch*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice);

	gT_FanVolumeTexture.addressMode[0] = cudaAddressModeBorder;
	gT_FanVolumeTexture.addressMode[1] = cudaAddressModeBorder;
	gT_FanVolumeTexture.filterMode = cudaFilterModeLinear;
	gT_FanVolumeTexture.normalized = false;

	// TODO: For very small sizes (roughly <=512x128) with few angles (<=180)
	// not using an array is more efficient.
	//cudaBindTexture2D(0, gT_FanVolumeTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);
	cudaBindTextureToArray(gT_FanVolumeTexture, dataArray, channelDesc);

	// TODO: error value?

	return true;
}
