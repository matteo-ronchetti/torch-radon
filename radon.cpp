#include <torch/extension.h>

#include <iostream>
#include <vector>

// CUDA forward declarations
void copy_image_cuda(float* x, float* y);
void radon_forward_cuda(const float* x, const float* rays, const float* angles, float* y, const int n_rays, const int n_angles);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// TODO remove
torch::Tensor radon_copy(torch::Tensor x) {
    CHECK_INPUT(x);
    auto y = torch::zeros_like(x);
    copy_image_cuda(x.data<float>(), y.data<float>());
    return y;
}

torch::Tensor radon_forward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles) {
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int n_rays = rays.size(0);
    const int n_angles = angles.size(0);
    
    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({n_angles, n_rays}, options);
    
    radon_forward_cuda(x.data<float>(), rays.data<float>(), angles.data<float>(), y.data<float>(), n_rays, n_angles);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy", &radon_copy, "Sample copy");
  m.def("forward", &radon_forward, "Radon forward");
}
