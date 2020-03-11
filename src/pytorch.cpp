#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "forward.h"
#include "backprojection.h"
#include "noise.h"
#include "texture.h"
#include "filtering.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor radon_forward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache &tex_cache){
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int batch_size = x.size(0);
    const int img_size = x.size(1);
    TORCH_CHECK(x.size(2) == img_size, "Images in x must be square")
    TORCH_CHECK(img_size % 16 == 0, "Size of images in x must be multiple of 16")

    const int n_rays = rays.size(0);
    const int n_angles = angles.size(0);
    const int device = x.device().index();

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, n_angles, n_rays}, options);

    radon_forward_cuda(x.data_ptr<float>(), rays.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                       tex_cache,
                       batch_size, img_size, n_rays, n_angles, device);

    return y;
}

torch::Tensor radon_backward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache &tex_cache, const bool extend) {
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int batch_size = x.size(0);
    const int n_angles = x.size(1);
    const int img_size = x.size(2);
    const int n_rays = rays.size(0);
    const int device = x.device().index();

    TORCH_CHECK(angles.size(0) == n_angles, "Mismatch between sinogram size and number of angles")
    TORCH_CHECK(img_size % 16 == 0, "Dimension 2 of sinogram (i.e. image size) must be multiple of 16")

    // create output image tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, img_size, img_size}, options);

    radon_backward_cuda(x.data_ptr<float>(), rays.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                        tex_cache,
                        batch_size, img_size, n_rays, n_angles, device, extend);

    return y;

}

torch::Tensor radon_backward_lb(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache &tex_cache, const bool extend) {
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int n_angles = x.size(0);
    const int img_size = x.size(1);
    const int batch_size = x.size(2);

    const int n_rays = rays.size(0);
    const int device = x.device().index();

    TORCH_CHECK(angles.size(0) == n_angles, "Mismatch between sinogram size and number of angles")
    TORCH_CHECK(img_size % 16 == 0, "Dimension 1 of sinogram (i.e. image size) must be multiple of 16")
    TORCH_CHECK(batch_size % 16 == 0, "Dimension 0 of sinogram (i.e. batch size) must be multiple of 16")

    // create output image tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, img_size, img_size}, options);

    radon_backward_cuda_lb(x.data_ptr<float>(), rays.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                           tex_cache,
                           batch_size, img_size, n_rays, n_angles, device, extend);

    return y;
}

void radon_add_noise(torch::Tensor x, RadonNoiseGenerator& noise_generator, const float signal,
                     const float density_normalization, const bool approximate) {
    CHECK_INPUT(x);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();


    noise_generator.add_noise(x.data_ptr<float>(), signal, density_normalization, approximate, width, height, device);
}

torch::Tensor emulate_sensor_readings(torch::Tensor x, RadonNoiseGenerator& noise_generator, const float signal,
                                      const float density_normalization) {
    CHECK_INPUT(x);

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto y = torch::empty({x.size(0), x.size(1), x.size(2)}, options);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    noise_generator.emulate_readings(x.data_ptr<float>(), y.data_ptr<int>(), signal, density_normalization, width,
                                     height, device);

    return y;
}

torch::Tensor readings_lookup(torch::Tensor x, torch::Tensor lookup_table) {
    CHECK_INPUT(x);
    CHECK_INPUT(lookup_table);
    TORCH_CHECK(x.dtype() == torch::kInt32, "Input tensor must have type Int32")

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({x.size(0), x.size(1), x.size(2)}, options);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    readings_lookup_cuda(x.data_ptr<int>(), y.data_ptr<float>(), lookup_table.data_ptr<float>(),
                         lookup_table.size(0), width, height, device);

    return y;
}

torch::Tensor radon_filter_sinogram(torch::Tensor x, FFTCache& fft_cache) {
    CHECK_INPUT(x);

    const int batch_size = x.size(0);
    const int n_angles = x.size(1);
    const int n_rays = x.size(2);
    const int device = x.device().index();

    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, n_angles, n_rays}, options);

    radon_filter_sinogram_cuda(x.data_ptr<float>(), y.data_ptr<float>(), fft_cache, batch_size, n_angles, n_rays, device);

    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &radon_forward, "Radon forward projection");
m.def("backward", &radon_backward, "Radon back projection");
m.def("backward_lb", &radon_backward_lb, "Radon back projection");
m.def("add_noise", &radon_add_noise, "Add noise to sinogram");
m.def("emulate_sensor_readings", &emulate_sensor_readings, "Emulate sensor readings");
m.def("readings_lookup", &readings_lookup, "Lookup sensors readings in a table");
m.def("filter_sinogram", &radon_filter_sinogram, "Apply filtering to a sinogram");

py::class_<TextureCache>(m,"TextureCache")
    .def (py::init<size_t>())
    .def("free", &TextureCache::free);

py::class_<FFTCache>(m,"FFTCache")
.def (py::init<size_t>())
.def("free", &FFTCache::free);

py::class_<RadonNoiseGenerator>(m, "RadonNoiseGenerator")
    .def(py::init<const uint>())
    .def("set_seed", (void (RadonNoiseGenerator::*)(const uint)) &RadonNoiseGenerator::set_seed)
    .def("free", &RadonNoiseGenerator::free);
}
