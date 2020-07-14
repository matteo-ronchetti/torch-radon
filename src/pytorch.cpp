#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "forward.h"
#include "backprojection.h"
#include "noise.h"
#include "texture.h"
#include "filtering.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor radon_forward(torch::Tensor x, const int det_count, const float det_spacing, torch::Tensor angles, TextureCache &tex_cache) {
    CHECK_INPUT(x);
    CHECK_INPUT(angles);
    //

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int img_size = x.size(1);
    const int n_angles = angles.size(0);
    const int device = x.device().index();

    TORCH_CHECK(x.size(2) == img_size, "Images in x must be square")
    TORCH_CHECK(img_size % 16 == 0, "Size of images in x must be multiple of 16")
    if (dtype == torch::kFloat16) {
        TORCH_CHECK(batch_size % 4 == 0, "Batch size must be multiple of 4 when using half precision")
    }

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, n_angles, det_count}, options);

    if (dtype == torch::kFloat16) {
        radon_forward_cuda((unsigned short *) x.data_ptr<at::Half>(), det_count, det_spacing, angles.data_ptr<float>(),
                           (unsigned short *) y.data_ptr<at::Half>(),
                           tex_cache,
                           batch_size, img_size, n_angles, device);
    } else {
        radon_forward_cuda(x.data_ptr<float>(), det_count, det_spacing, angles.data_ptr<float>(), y.data_ptr<float>(),
                           tex_cache,
                           batch_size, img_size, n_angles, device);
    }
    return y;
}

torch::Tensor
radon_backward(torch::Tensor x, const int det_count, const float det_spacing, torch::Tensor angles, TextureCache &tex_cache) {
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int n_angles = x.size(1);
    const int img_size = x.size(2);
    const int device = x.device().index();

    TORCH_CHECK(angles.size(0) == n_angles, "Mismatch between sinogram size and number of angles")
    TORCH_CHECK(angles.size(0) <= 512, "Can only support up to 512 angles")
    TORCH_CHECK(img_size % 16 == 0, "Dimension 2 of sinogram (i.e. image size) must be multiple of 16")
    if (dtype == torch::kFloat16) {
        TORCH_CHECK(batch_size % 4 == 0, "Batch size must be multiple of 4 when using half precision")
    }

    // create output image tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, img_size, img_size}, options);

    if (dtype == torch::kFloat16) {
        radon_backward_cuda((unsigned short *) x.data_ptr<at::Half>(), det_count, det_spacing, angles.data_ptr<float>(),
                            (unsigned short *) y.data_ptr<at::Half>(),
                            tex_cache, batch_size, img_size, n_angles, device);
    } else {
        radon_backward_cuda(x.data_ptr<float>(), det_count, det_spacing, angles.data_ptr<float>(), y.data_ptr<float>(),
                            tex_cache, batch_size, img_size, n_angles, device);
    }


    return y;

}

void radon_add_noise(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
                     const float density_normalization, const bool approximate) {
    CHECK_INPUT(x);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();


    noise_generator.add_noise(x.data_ptr<float>(), signal, density_normalization, approximate, width, height, device);
}

torch::Tensor emulate_sensor_readings(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
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

torch::Tensor torch_emulate_readings_new(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
                                         const float normal_std, const int k, const int bins) {
    CHECK_INPUT(x);

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto y = torch::empty({x.size(0), x.size(1), x.size(2)}, options);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    noise_generator.emulate_readings_new(x.data_ptr<float>(), y.data_ptr<int>(), signal, normal_std, k, bins, width,
                                         height, device);

    return y;
}

torch::Tensor emulate_readings_multilevel(torch::Tensor x, RadonNoiseGenerator &noise_generator, torch::Tensor signal,
                                          torch::Tensor normal_std, torch::Tensor k, torch::Tensor levels,
                                          const int bins) {
    CHECK_INPUT(x);
    CHECK_INPUT(signal);
    CHECK_INPUT(normal_std);
    CHECK_INPUT(k);
    CHECK_INPUT(levels);
    TORCH_CHECK(k.dtype() == torch::kInt32, "Input tensor 'k' must have type Int32");
    TORCH_CHECK(levels.dtype() == torch::kInt32, "Input tensor 'levels' must have type Int32");
    TORCH_CHECK(x.size(0) == levels.size(0), "Size of 'levels' must be equal to batch size");


    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto y = torch::empty({x.size(0), x.size(1), x.size(2)}, options);

    const int batch = x.size(0);
    const int height = x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    noise_generator.emulate_readings_multilevel(x.data_ptr<float>(), y.data_ptr<int>(), signal.data_ptr<float>(),
                                                normal_std.data_ptr<float>(), k.data_ptr<int>(), levels.data_ptr<int>(),
                                                bins, batch, width, height, device);

    return y;
}

torch::Tensor readings_lookup(torch::Tensor x, torch::Tensor lookup_table) {
    CHECK_INPUT(x);
    CHECK_INPUT(lookup_table);
    TORCH_CHECK(x.dtype() == torch::kInt32, "Input tensor must have type Int32");

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

torch::Tensor readings_lookup_multilevel(torch::Tensor x, torch::Tensor lookup_table, torch::Tensor levels) {
    CHECK_INPUT(x);
    CHECK_INPUT(lookup_table);
    CHECK_INPUT(levels);
    TORCH_CHECK(x.dtype() == torch::kInt32, "Input tensor 'x' must have type Int32");
    TORCH_CHECK(levels.dtype() == torch::kInt32, "Input tensor 'levels' must have type Int32");

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({x.size(0), x.size(1), x.size(2)}, options);

    const int batch = x.size(0);
    const int height = x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    readings_lookup_multilevel_cuda(x.data_ptr<int>(), y.data_ptr<float>(), lookup_table.data_ptr<float>(),
                                    levels.data_ptr<int>(),
                                    lookup_table.size(1), batch, width, height, device);

    return y;
}

torch::Tensor radon_filter_sinogram(torch::Tensor x, FFTCache &fft_cache) {
    CHECK_INPUT(x);

    const int batch_size = x.size(0);
    const int n_angles = x.size(1);
    const int n_rays = x.size(2);
    const int device = x.device().index();

    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({batch_size, n_angles, n_rays}, options);

    radon_filter_sinogram_cuda(x.data_ptr<float>(), y.data_ptr<float>(), fft_cache, batch_size, n_angles, n_rays,
                               device);

    return y;
}

std::pair<float, float> torch_compute_ab(torch::Tensor x, const float signal, const float eps, const int k) {
    CHECK_INPUT(x);

    const int device = x.device().index();

    return compute_ab(x.data_ptr<float>(), x.size(0), signal, eps, k, device);
}

std::pair <torch::Tensor, torch::Tensor>
torch_compute_lookup_table(torch::Tensor x, torch::Tensor weights, const float signal, const int bins, const int scale,
                           torch::Tensor log_factorial, torch::Tensor border_w) {
    CHECK_INPUT(x);
    CHECK_INPUT(weights);
    TORCH_CHECK(weights.size(0) <= 256, "weights can have max 256 elements");
    TORCH_CHECK(border_w.size(0) == scale, "border_w must have length == scale");

    const int device = x.device().index();

    // create output sinogram tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto y_mean = torch::empty({bins}, options);
    auto y_var = torch::empty({bins}, options);

    compute_lookup_table(x.data_ptr<float>(), weights.data_ptr<float>(), y_mean.data_ptr<float>(),
                         y_var.data_ptr<float>(), log_factorial.data_ptr<float>(), border_w.data_ptr<float>(),
                                 x.size(0), weights.size(0), signal, bins, scale, device);

    return make_pair(y_mean, y_var);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &radon_forward, "Radon forward projection");

m.def("backward", &radon_backward, "Radon back projection");
//m.def("backward_lb", &radon_backward_lb, "Radon back projection");
m.def("add_noise", &radon_add_noise, "Add noise to sinogram");
m.def("emulate_sensor_readings", &emulate_sensor_readings, "Emulate sensor readings");
m.def("readings_lookup", &readings_lookup, "Lookup sensors readings in a table");
m.def("filter_sinogram", &radon_filter_sinogram, "Apply filtering to a sinogram");
m.def("compute_ab", &torch_compute_ab, "TODO");
m.def("compute_lookup_table", &torch_compute_lookup_table, "TODO");
m.def("emulate_readings_new", &torch_emulate_readings_new, "TODO");
m.def("emulate_readings_multilevel", &emulate_readings_multilevel, "TODO");
m.def("readings_lookup_multilevel", &readings_lookup_multilevel, "TODO");


py::class_<TextureCache>(m,
"TextureCache")
.

def (py::init<size_t>())

.def("free", &TextureCache::free);

py::class_<FFTCache>(m,
"FFTCache")
.

def (py::init<size_t>())

.def("free", &FFTCache::free);

py::class_<RadonNoiseGenerator>(m,
"RadonNoiseGenerator")
.

def (py::init<const uint>())

.def("set_seed", (void (RadonNoiseGenerator::*)(const uint)) &RadonNoiseGenerator::set_seed)
.def("free", &RadonNoiseGenerator::free);
}
