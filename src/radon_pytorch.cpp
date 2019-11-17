#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "radon_forward.h"
#include "radon_backprojection.h"
#include "radon_noise.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor radon_forward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache tex_cache) {
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int batch_size = x.size(0);
    const int img_size = x.size(1);
    TORCH_CHECK(x.size(2) == img_size, "Images in x must be squared")

    const int n_rays = rays.size(0);
    const int n_angles = angles.size(0);

    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, n_angles, n_rays}, options);

    radon_forward_cuda(x.data<float>(), rays.data<float>(), angles.data<float>(), y.data<float>(), tex_cache,
                       batch_size, img_size, n_rays, n_angles);

    return y;
}

torch::Tensor radon_backward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache tex_cache) {
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int batch_size = x.size(0);
    const int n_angles = x.size(1);
    const int img_size = x.size(2);
    const int n_rays = rays.size(0);

    TORCH_CHECK(angles.size(0) == n_angles, "Radon backward mismatch between sinogram size and number of angles")

    // create output image tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, img_size, img_size}, options);

    radon_backward_cuda(x.data<float>(), rays.data<float>(), angles.data<float>(), y.data<float>(), tex_cache,
                       batch_size, img_size, n_rays, n_angles);

    return y;
}

void radon_add_noise(torch::Tensor x, RadonNoiseGenerator noise_generator, const float signal, const float density_normalization, const bool approximate) {
    CHECK_INPUT(x);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);

    noise_generator.add_noise(x.data<float>(), signal, density_normalization, approximate, width, height);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &radon_forward, "Radon forward projection");
  m.def("backward", &radon_backward, "Radon backprojection");
  m.def("add_noise", &radon_add_noise, "Add noise to sinogram");


  py::class_<TextureCache>(m, "TextureCache")
      .def(py::init<>())
      .def("free", &TextureCache::free);

  py::class_<RadonNoiseGenerator>(m, "RadonNoiseGenerator")
      .def(py::init<const uint>())
      .def("set_seed", &RadonNoiseGenerator::set_seed)
      .def("free", &RadonNoiseGenerator::free);

  //m.def("filter_sinogram", &radon_filter_sinogram, "Radon backprojection");

}
