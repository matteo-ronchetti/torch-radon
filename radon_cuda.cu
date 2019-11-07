#include <iostream>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void radon_forward_kernel(float* output, cudaTextureObject_t texObj, const float* rays, const float* angles, const int img_size, const int n_rays, const int n_angles) {
    // Calculate texture coordinates
    const uint ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint batch_id = blockIdx.y;
    const float rsx = rays[ray_id*4+0];
    const float rsy = rays[ray_id*4+1];
    const float rex = rays[ray_id*4+2];
    const float rey = rays[ray_id*4+3];
    const float v = img_size/2;

    const float vx = (rex-rsx)/img_size;
    const float vy = (rey-rsy)/img_size;
    const float n = hypot(vx, vy);

    for(int i = 0; i < n_angles; i++){
        // rotate ray
        float angle = angles[i];
        float cs = __cosf(angle);
        float sn = __sinf(angle);

        float sx = rsx*cs - rsy*sn + v;
        float sy = rsx*sn + rsy*cs + v;
        float rvx = vx*cs - vy*sn;
        float rvy = vx*sn + vy*cs;

        float tmp = 0.0;
        for(int j = 0; j < img_size; j++){
            tmp += tex2DLayered<float>(texObj, sx+rvx*j, sy+rvy*j, batch_id);
        }

        output[batch_id*n_rays*n_angles + i*n_rays + ray_id] = tmp*n;
    }
}

void radon_forward_cuda(const float* x, const float* rays, const float* angles, float* y, const int batch_size, const int img_size, const int n_rays, const int n_angles){
    cudaArray* tmp;
    auto my_tex = create_texture(x, tmp, batch_size, img_size, img_size, img_size);

    // Invoke kernel
    dim3 dimGrid(8, batch_size);
    dim3 dimBlock(16);

    radon_forward_kernel<<<dimGrid, dimBlock>>>(y, my_tex, rays, angles, img_size, n_rays, n_angles);

    cudaFreeArray(tmp);
}

__global__ void radon_backward_kernel(float* output, cudaTextureObject_t texObj, const float* rays, const float* angles, const int img_size, const int n_rays, const int n_angles) {
    // Calculate texture coordinates
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint batch_id = blockIdx.z;
    const float dx = (float)x - img_size/2  + 0.5;
    const float dy = (float)y - img_size/2  + 0.5;
    const float v = img_size/2;
    float tmp = 0.0;

    for(int i = 0; i < n_angles; i++){
        // TODO cache angles
        float angle = angles[i];
        float j = __cosf(angle) * dx + __sinf(angle) * dy + v;
        tmp += tex2DLayered<float>(texObj, j, i+0.5f, batch_id);
    }

    output[batch_id*img_size*img_size + y*img_size + x] = tmp;
}

void radon_backward_cuda(const float* x, const float* rays, const float* angles, float* y, const int batch_size, const int img_size, const int n_rays, const int n_angles){
    //std::cout << batch_size << " " << img_size << " " << n_rays << " " << n_angles << std::endl;
    cudaArray* tmp;
    auto my_tex = create_texture(x, tmp, batch_size, n_rays, n_angles, n_rays);

    // Invoke kernel
    dim3 dimGrid(8, 8, batch_size);
    dim3 dimBlock(16, 16);

    radon_backward_kernel<<<dimGrid, dimBlock>>>(y, my_tex, rays, angles, img_size, n_rays, n_angles);

    cudaFreeArray(tmp);
}


#define MIN(a,b) (((a)<(b))?(a):(b))
__global__ void apply_filter(cufftComplex* sino,const float* f) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    const int numSensors = 256;
    const float k = MIN(float(x), 256.0f - float(x)) / (float)numSensors;

    sino[numSensors*y + x].x *= k;
    sino[numSensors*y + x].y *= k;
}


void radon_filter_sinogram_cuda(const float* x, const float* f, float* y, const int batch_size, const int n_rays, const int n_angles){
    //TODO int padded_size = next_power_of_two(n_rays);
    int padded_size = n_rays*2;
    cufftReal* padded_data = nullptr;// (cufftReal*) x;
    checkCudaErrors(cudaMalloc((void**)&padded_data, sizeof(cufftReal)*batch_size*n_angles*padded_size));
    checkCudaErrors(cudaMemcpy2D(padded_data, sizeof(cufftReal)*padded_size, x, sizeof(float)*n_rays, sizeof(float)*n_rays, batch_size*n_angles, cudaMemcpyDeviceToDevice));


    cufftComplex* complex_data = nullptr;
    checkCudaErrors(cudaMalloc((void**)&complex_data, sizeof(cufftComplex)*batch_size*n_angles*padded_size));

    cufftHandle forward_plan, back_plan;
    checkCudaErrors(cufftPlan1d(&forward_plan, padded_size, CUFFT_R2C, batch_size*n_angles));
    checkCudaErrors(cufftPlan1d(&back_plan, padded_size, CUFFT_C2R, batch_size*n_angles));

    checkCudaErrors(cufftExecR2C(forward_plan, padded_data, complex_data));

    // TODO è fatto a cazzo
    apply_filter<<<dim3(16, n_angles), dim3(16, 16)>>>(complex_data, f);

    // TODO pad also y
    checkCudaErrors(cufftExecC2R(back_plan, complex_data, y));
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cufftDestroy(forward_plan));
    checkCudaErrors(cufftDestroy(back_plan));
}

int main(){
    std::cout << "float: " << sizeof(float) << std::endl;
    std::cout << "cufftReal: " << sizeof(cufftReal) << std::endl;
    std::cout << "cufftComplex: " << sizeof(cufftComplex) << std::endl;
    std::cout << "Hello CUDA" << std::endl;
}
