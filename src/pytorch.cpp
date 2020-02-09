#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "forward.h"
#include "backprojection.h"
#include "noise.h"
#include "filtering.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor sinofilter(torch::Tensor x, torch::Tensor w){
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int img_size = x.size(2);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::zeros({batch_size, channels, img_size, img_size}, options);

    sinofilter_cuda(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), batch_size, channels, img_size);
    return y;
}


torch::Tensor radon_forward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache &tex_cache) {
    CHECK_INPUT(x);
    CHECK_INPUT(rays);
    CHECK_INPUT(angles);

    const int batch_size = x.size(0);
    const int img_size = x.size(1);
    TORCH_CHECK(x.size(2) == img_size, "Images in x must be squared")

    const int n_rays = rays.size(0);
    const int n_angles = angles.size(0);
    std::cout << x.device().index() << std::endl;

    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, n_angles, n_rays}, options);

    radon_forward_cuda(x.data_ptr<float>(), rays.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                       tex_cache,
                       batch_size, img_size, n_rays, n_angles);

    return y;
}

torch::Tensor radon_backward(torch::Tensor x, torch::Tensor rays, torch::Tensor angles, TextureCache &tex_cache) {
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

    radon_backward_cuda(x.data_ptr<float>(), rays.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                        tex_cache,
                        batch_size, img_size, n_rays, n_angles);

    return y;
}

void radon_add_noise(torch::Tensor x, RadonNoiseGenerator noise_generator, const float signal,
                     const float density_normalization, const bool approximate) {
    CHECK_INPUT(x);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);

    noise_generator.add_noise(x.data_ptr<float>(), signal, density_normalization, approximate, width, height);
}

torch::Tensor emulate_sensor_readings(torch::Tensor x, RadonNoiseGenerator noise_generator, const float signal,
                                      const float density_normalization) {
    CHECK_INPUT(x);

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto y = torch::empty({x.size(0), x.size(1), x.size(2)}, options);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);

    noise_generator.emulate_readings(x.data_ptr<float>(), y.data_ptr<int>(), signal, density_normalization, width,
                                     height);

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

    readings_lookup_cuda(x.data_ptr<int>(), y.data_ptr<float>(), lookup_table.data_ptr<float>(),
                         lookup_table.size(0), width, height);

    return y;
}

torch::Tensor radon_filter_sinogram(torch::Tensor x) {
    CHECK_INPUT(x);

    const int batch_size = x.size(0);
    const int n_angles = x.size(1);
    const int n_rays = x.size(2);

    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, n_angles, n_rays}, options);

    radon_filter_sinogram_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, n_angles, n_rays);

    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &radon_forward, "Radon forward projection");
m.def("backward", &radon_backward, "Radon backprojection");
m.def("add_noise", &radon_add_noise, "Add noise to sinogram");
m.def("emulate_sensor_readings", &emulate_sensor_readings, "Emulate sensor readings");
m.def("readings_lookup", &readings_lookup, "Emulate sensor readings");
m.def("filter_sinogram", &radon_filter_sinogram, "Radon backprojection");
m.def("sinofilter", &sinofilter, "Radon backprojection");

py::class_<TextureCache>(m,"TextureCache")
    .def (py::init<>())
    .def("free", &TextureCache::free);

py::class_<RadonNoiseGenerator>(m, "RadonNoiseGenerator")
    .def(py::init<const uint>())
    .def("set_seed", &RadonNoiseGenerator::set_seed)
    .def("free", &RadonNoiseGenerator::free);
}
