#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <math.h>

#include "parameter_classes.h"
#include "forward.h"
#include "backprojection.h"
#include "noise.h"
#include "texture.h"
#include "utils.h"
#include "symbolic.h"
#include "log.h"
#include "fft.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor torch_symbolic_forward(const SymbolicFunction &f, torch::Tensor angles, const ProjectionCfg &proj)
{
    TORCH_CHECK(!angles.device().is_cuda(), "angles must be on CPU");
    CHECK_CONTIGUOUS(angles);

    const int n_angles = angles.size(0);
    auto y = torch::empty({n_angles, proj.det_count_u});

    symbolic_forward(f, proj, angles.data_ptr<float>(), n_angles, y.data_ptr<float>());

    return y;
}

torch::Tensor torch_symbolic_discretize(const SymbolicFunction &f, const int height, const int width)
{
    auto y = torch::empty({height, width});

    f.discretize(y.data_ptr<float>(), height, width);

    return y;
}

torch::Tensor radon_forward(torch::Tensor x, torch::Tensor angles, TextureCache &tex_cache,
                            const VolumeCfg vol_cfg, const ProjectionCfg proj_cfg, const ExecCfg exec_cfg)
{
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int n_angles = angles.size(0);
    const int device = x.device().index();

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());

    if (vol_cfg.is_3d)
    {
        auto y = torch::empty({batch_size, n_angles, proj_cfg.det_count_v, proj_cfg.det_count_u}, options);

        if (dtype == torch::kFloat16)
        {
            radon_forward_cuda_3d((unsigned short *)x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                                  (unsigned short *)y.data_ptr<at::Half>(),
                                  tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        else
        {
            radon_forward_cuda_3d(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                                  tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        return y;
    }
    else
    {
        auto y = torch::empty({batch_size, n_angles, proj_cfg.det_count_u}, options);

        if (dtype == torch::kFloat16)
        {
            radon_forward_cuda((unsigned short *)x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                               (unsigned short *)y.data_ptr<at::Half>(),
                               tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        else
        {
            radon_forward_cuda(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                               tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        return y;
    }
}

torch::Tensor
radon_backward(torch::Tensor x, torch::Tensor angles, TextureCache &tex_cache, const VolumeCfg vol_cfg,
               const ProjectionCfg proj_cfg, const ExecCfg exec_cfg)
{
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int device = x.device().index();

    TORCH_CHECK(angles.size(0) <= 4096, "Can only support up to 4096 angles")

    // create output image tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());

    if (vol_cfg.is_3d)
    {
        auto y = torch::empty({batch_size, vol_cfg.depth, vol_cfg.height, vol_cfg.width}, options);

        if (dtype == torch::kFloat16)
        {
            radon_backward_cuda_3d((unsigned short *)x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                                   (unsigned short *)y.data_ptr<at::Half>(),
                                   tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        else
        {
            radon_backward_cuda_3d(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                                   tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        return y;
    }
    else
    {
        auto y = torch::empty({batch_size, vol_cfg.height, vol_cfg.width}, options);

        if (dtype == torch::kFloat16)
        {
            radon_backward_cuda((unsigned short *)x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                                (unsigned short *)y.data_ptr<at::Half>(),
                                tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }
        else
        {
            radon_backward_cuda(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                                tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
        }

        return y;
    }
}

void radon_add_noise(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
                     const float density_normalization, const bool approximate)
{
    CHECK_INPUT(x);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    noise_generator.add_noise(x.data_ptr<float>(), signal, density_normalization, approximate, width, height, device);
}

torch::Tensor emulate_sensor_readings(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
                                      const float density_normalization)
{
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

torch::Tensor torch_rfft(torch::Tensor x, FFTCache &fft_cache)
{
    CHECK_INPUT(x);

    const int device = x.device().index();
    const int rows = x.size(0) * x.size(1);
    const int cols = x.size(2);

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    auto y = torch::empty({x.size(0), x.size(1), x.size(2) / 2 + 1, 2}, options);

    FFT(fft_cache, x.data_ptr<float>(), device, rows, cols, y.data_ptr<float>());

    return y;
}

torch::Tensor torch_irfft(torch::Tensor x, FFTCache &fft_cache)
{
    CHECK_INPUT(x);

    const int device = x.device().index();
    const int rows = x.size(0) * x.size(1);
    const int cols = (x.size(2) - 1) * 2;

    // create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    auto y = torch::empty({x.size(0), x.size(1), cols}, options);

    iFFT(fft_cache, x.data_ptr<float>(), device, rows, cols, y.data_ptr<float>());

    return y;
}

torch::Tensor torch_emulate_readings_new(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
                                         const float normal_std, const int k, const int bins)
{
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
                                          const int bins)
{
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

torch::Tensor readings_lookup(torch::Tensor x, torch::Tensor lookup_table)
{
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

torch::Tensor readings_lookup_multilevel(torch::Tensor x, torch::Tensor lookup_table, torch::Tensor levels)
{
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

std::pair<float, float> torch_compute_ab(torch::Tensor x, const float signal, const float eps, const int k)
{
    CHECK_INPUT(x);

    const int device = x.device().index();

    return compute_ab(x.data_ptr<float>(), x.size(0), signal, eps, k, device);
}

std::pair<torch::Tensor, torch::Tensor>
torch_compute_lookup_table(torch::Tensor x, torch::Tensor weights, const float signal, const int bins, const int scale,
                           torch::Tensor log_factorial, torch::Tensor border_w)
{
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &radon_forward, "Radon forward projection");
    m.def("backward", &radon_backward, "Radon back projection");

    m.def("add_noise", &radon_add_noise, "Add noise to sinogram");
    m.def("emulate_sensor_readings", &emulate_sensor_readings, "Emulate sensor readings");
    m.def("readings_lookup", &readings_lookup, "Lookup sensors readings in a table");
    m.def("compute_ab", &torch_compute_ab, "TODO");
    m.def("compute_lookup_table", &torch_compute_lookup_table, "TODO");
    m.def("emulate_readings_new", &torch_emulate_readings_new, "TODO");
    m.def("emulate_readings_multilevel", &emulate_readings_multilevel, "TODO");
    m.def("readings_lookup_multilevel", &readings_lookup_multilevel, "TODO");
    m.def("symbolic_forward", &torch_symbolic_forward, "TODO");
    m.def("symbolic_discretize", &torch_symbolic_discretize, "TODO");

    m.def("rfft", &torch_rfft, "TODO");
    m.def("irfft", &torch_irfft, "TODO");

    m.def("set_log_level", [](const int level) { Log::log_level = static_cast<Log::Level>(level); });

    py::class_<TextureCache>(m, "TextureCache")
        .def(py::init<size_t>())
        .def("free", &TextureCache::free);

    py::class_<FFTCache>(m, "FFTCache")
        .def(py::init<size_t>())
        .def("free", &FFTCache::free);

    py::class_<RadonNoiseGenerator>(m, "RadonNoiseGenerator")
        .def(py::init<const uint>())
        .def("set_seed", (void (RadonNoiseGenerator::*)(const uint)) & RadonNoiseGenerator::set_seed)
        .def("free", &RadonNoiseGenerator::free);

    py::class_<VolumeCfg>(m, "VolumeCfg")
        .def(py::init<int, int, int, float, float, float, float, float, float, bool>())
        .def_readonly("depth", &VolumeCfg::depth)
        .def_readonly("height", &VolumeCfg::height)
        .def_readonly("width", &VolumeCfg::width)
        .def_readonly("dx", &VolumeCfg::dx)
        .def_readonly("dy", &VolumeCfg::dy)
        .def_readonly("dz", &VolumeCfg::dz)
        .def_readonly("is_3d", &VolumeCfg::is_3d);

    py::class_<ProjectionCfg>(m, "ProjectionCfg")
        .def(py::init<int, float>())
        .def(py::init<int, float, int, float, float, float, float, float, int>())
        .def("is_2d", &ProjectionCfg::is_2d)
        .def("copy", &ProjectionCfg::copy)
        .def_readonly("projection_type", &ProjectionCfg::projection_type)
        .def_readwrite("det_count_u", &ProjectionCfg::det_count_u)
        .def_readwrite("det_spacing_u", &ProjectionCfg::det_spacing_u)
        .def_readwrite("det_count_v", &ProjectionCfg::det_count_v)
        .def_readwrite("det_spacing_v", &ProjectionCfg::det_spacing_v)
        .def_readwrite("s_dist", &ProjectionCfg::s_dist)
        .def_readwrite("d_dist", &ProjectionCfg::d_dist)
        .def_readwrite("pitch", &ProjectionCfg::pitch)
        .def_readwrite("initial_z", &ProjectionCfg::initial_z)
        .def_readwrite("n_angles", &ProjectionCfg::n_angles);

    py::class_<ExecCfg>(m, "ExecCfg")
        .def(py::init<int, int, int, int>());

    py::class_<SymbolicFunction>(m, "SymbolicFunction")
        .def(py::init<float, float>())
        .def("add_gaussian", &SymbolicFunction::add_gaussian)
        .def("add_ellipse", &SymbolicFunction::add_ellipse)
        .def("move", &SymbolicFunction::move)
        .def("scale", &SymbolicFunction::scale);
}
