#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <pybind11/numpy.h>

#include "parameter_classes.h"
#include "forward.h"
//#include "backprojection.h"
#include "noise.h"
#include "texture.h"
#include "utils.h"
#include "symbolic.h"
#include "log.h"
#include "fft.h"
#include "torch_radon.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

namespace py = pybind11;

typedef py::array_t<float, py::array::c_style> Array;

torch::Tensor torch_symbolic_forward(const SymbolicFunction &f, torch::Tensor angles, const Projection2D &proj)
{
    TORCH_CHECK(!angles.device().is_cuda(), "angles must be on CPU");
    CHECK_CONTIGUOUS(angles);

    const int n_angles = angles.size(0);
    auto y = torch::empty({n_angles, proj.det_count});

    symbolic_forward(f, proj, angles.data_ptr<float>(), n_angles, y.data_ptr<float>());

    return y;
}

torch::Tensor torch_symbolic_discretize(const SymbolicFunction &f, const int height, const int width)
{
    auto y = torch::empty({height, width});

    f.discretize(y.data_ptr<float>(), height, width);

    return y;
}

torch::Tensor radon_forward(const torch::Tensor &x, const torch::Tensor &angles, TextureCache &tex_cache,
                            const VolumeCfg &vol_cfg, const Projection2D &proj_cfg, const ExecCfg &exec_cfg)
{
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int n_angles = angles.size(0);
    const int device = x.device().index();

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, n_angles, proj_cfg.det_count}, options);

    if (dtype == torch::kFloat16)
    {
        radon_forward_cuda((ushort *)x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                           (ushort *)y.data_ptr<at::Half>(),
                           tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
    }
    else
    {
        radon_forward_cuda(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                           tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
    }
    return y;
}

torch::Tensor radon_forward(const torch::Tensor &x, const torch::Tensor &angles, TextureCache &tex_cache,
                            const VolumeCfg &vol_cfg, Projection3D &proj_cfg, const ExecCfg &exec_cfg)
{
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int n_angles = angles.size(0);
    const int device = x.device().index();

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, n_angles, proj_cfg.det_count_v, proj_cfg.det_count_u}, options);

    if (dtype == torch::kFloat16)
    {
        radon_forward_cuda_3d((ushort *)x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                              (ushort *)y.data_ptr<at::Half>(),
                              tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
    }
    else
    {
        radon_forward_cuda_3d(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                              tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
    }

    return y;
}

torch::Tensor radon_forward_batch(const torch::Tensor &x, const torch::Tensor &angles, TextureCache &tex_cache,
                                  const VolumeCfg &vol_cfg, std::vector<Projection3D> &proj_cfgs, const ExecCfg &exec_cfg)
{
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int n_angles = angles.size(0);
    const int device = x.device().index();

    TORCH_CHECK(n_angles == int(proj_cfgs.size()), "Number of angles must be the same as number of projections");

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, n_angles, proj_cfgs[0].det_count_v, proj_cfgs[0].det_count_u}, options);

    radon_forward_cuda_3d_batch(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                                tex_cache, vol_cfg, proj_cfgs, exec_cfg, batch_size, device);

    return y;
}

Array compute_projection_matrices(const Array &angles, const VolumeCfg &vol_cfg, std::vector<Projection3D> &proj_cfgs)
{
    py::buffer_info buff = angles.request();
    int n_angles = buff.shape[0];
    TORCH_CHECK(n_angles == int(proj_cfgs.size()), "Number of angles must be the same as number of projections");
    float *angles_data = static_cast<float *>(buff.ptr);
    py::array_t<float> res({n_angles, 3, 4});
    float *res_data = static_cast<float *>(res.request().ptr);

    projection_matrices(res_data, angles_data, vol_cfg, proj_cfgs);

    return res;
}

//
// torch::Tensor radon_backward(torch::Tensor x, torch::Tensor angles, TextureCache &tex_cache, const VolumeCfg &vol_cfg,
//               const Projection &proj_cfg, const ExecCfg &exec_cfg) {
//    CHECK_INPUT(x);
//    CHECK_INPUT(angles);
//
//    auto dtype = x.dtype();
//
//    const int batch_size = x.size(0);
//    const int device = x.device().index();
//
//    TORCH_CHECK(angles.size(0) <= 4096, "Can only support up to 4096 angles")
//
//    // create output image tensor
//    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
//
//    if (vol_cfg.is_3d) {
//        auto y = torch::empty({batch_size, vol_cfg.depth, vol_cfg.height, vol_cfg.width}, options);
//
//        if (dtype == torch::kFloat16) {
//            radon_backward_cuda_3d((ushort *) x.data_ptr<at::Half>(), angles.data_ptr<float>(),
//                                   (ushort *) y.data_ptr<at::Half>(),
//                                   tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
//        } else {
//            radon_backward_cuda_3d(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
//                                   tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
//        }
//        return y;
//    } else {
//        auto y = torch::empty({batch_size, vol_cfg.height, vol_cfg.width}, options);
//
//        if (dtype == torch::kFloat16) {
//            radon_backward_cuda((ushort *) x.data_ptr<at::Half>(), angles.data_ptr<float>(),
//                                (ushort *) y.data_ptr<at::Half>(),
//                                tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
//        } else {
//            radon_backward_cuda(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
//                                tex_cache, vol_cfg, proj_cfg, exec_cfg, batch_size, device);
//        }
//
//        return y;
//    }
//}

void radon_add_noise(torch::Tensor x, RadonNoiseGenerator &noise_generator, const float signal,
                     const float density_normalization, const bool approximate)
{
    CHECK_INPUT(x);

    const int height = x.size(0) * x.size(1);
    const int width = x.size(2);
    const int device = x.device().index();

    noise_generator.add_noise(x.data_ptr<float>(), signal, density_normalization, approximate, width, height, device);
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

py::array_t<float> poseToNumpy(const pose &m)
{
    py::array_t<float> arr({4, 4});
    arr.mutable_at(0, 0) = m.x.x;
    arr.mutable_at(0, 1) = m.x.y;
    arr.mutable_at(0, 2) = m.x.z;
    arr.mutable_at(0, 3) = m.d.x;
    arr.mutable_at(1, 0) = m.y.x;
    arr.mutable_at(1, 1) = m.y.y;
    arr.mutable_at(1, 2) = m.y.z;
    arr.mutable_at(1, 3) = m.d.y;
    arr.mutable_at(2, 0) = m.z.x;
    arr.mutable_at(2, 1) = m.z.y;
    arr.mutable_at(2, 2) = m.z.z;
    arr.mutable_at(2, 3) = m.d.z;
    arr.mutable_at(3, 0) = 0.0f;
    arr.mutable_at(3, 1) = 0.0f;
    arr.mutable_at(3, 2) = 0.0f;
    arr.mutable_at(3, 3) = 1.0f;
    return arr;
}

void numpyToPose(pose &m, const py::array_t<float> &arr)
{
    m.x = {arr.at<float>(0, 0), arr.at<float>(0, 1), arr.at<float>(0, 2)};
    m.y = {arr.at<float>(1, 0), arr.at<float>(1, 1), arr.at<float>(1, 2)};
    m.z = {arr.at<float>(2, 0), arr.at<float>(2, 1), arr.at<float>(2, 2)};
    m.d = {arr.at<float>(0, 3), arr.at<float>(1, 3), arr.at<float>(2, 3)};
}

// @formatter:off
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", py::overload_cast<const torch::Tensor &, const torch::Tensor &, TextureCache &, const VolumeCfg &, const Projection2D &, const ExecCfg &>(&radon_forward), "Radon forward projection");
    m.def("forward", py::overload_cast<const torch::Tensor &, const torch::Tensor &, TextureCache &, const VolumeCfg &, Projection3D &, const ExecCfg &>(&radon_forward), "Radon forward projection");
    m.def("forward_batch", &radon_forward_batch, "Radon forward batch projection");
    m.def("projection_matrices", &compute_projection_matrices, "TODO");
    // m.def("backward", &radon_backward, "Radon back projection");

    m.def("add_noise", &radon_add_noise, "Add noise to sinogram");

    m.def("symbolic_forward", &torch_symbolic_forward, "TODO");
    m.def("symbolic_discretize", &torch_symbolic_discretize, "TODO");

    m.def("rfft", &torch_rfft, "TODO");
    m.def("irfft", &torch_irfft, "TODO");

    m.def("set_log_level", [](const int level)
          { Log::log_level = static_cast<Log::Level>(level); });

    py::enum_<ProjectionType>(m, "ProjectionType")
        .value("ParallelBeam", ProjectionType::ParallelBeam)
        .value("FanBeam", ProjectionType::FanBeam)
        .value("ConeBeam", ProjectionType::ConeBeam)
        .export_values();

    py::class_<TextureCache>(m, "TextureCache").def(py::init<size_t>()).def("free", &TextureCache::free);

    py::class_<FFTCache>(m, "FFTCache").def(py::init<size_t>()).def("free", &FFTCache::free);

    py::class_<RadonNoiseGenerator>(m, "RadonNoiseGenerator")
        .def(py::init<const uint>())
        .def("set_seed", (void(RadonNoiseGenerator::*)(const uint)) & RadonNoiseGenerator::set_seed)
        .def("free", &RadonNoiseGenerator::free);

    py::class_<VolumeCfg>(m, "VolumeCfg")
        .def(py::init<int, int, int, float, float, float>())
        .def_readonly("slices", &VolumeCfg::slices)
        .def_readonly("height", &VolumeCfg::height)
        .def_readonly("width", &VolumeCfg::width)
        .def("is_3d", &VolumeCfg::is_3d);

    py::class_<Projection2D>(m, "Projection2D")
        .def("ParallelBeam", &Projection2D::ParallelBeam)
        .def("FanBeam", &Projection2D::FanBeam)
        .def_readonly("type", &Projection2D::type)
        .def_readwrite("det_count", &Projection2D::det_count)
        .def_readwrite("det_spacing", &Projection2D::det_spacing)
        .def_readwrite("s_dist", &Projection2D::s_dist)
        .def_readwrite("d_dist", &Projection2D::d_dist)
        .def_readwrite("n_angles", &Projection2D::n_angles);

    py::class_<Projection3D>(m, "Projection3D")
        .def("ConeBeam", &Projection3D::ConeBeam)
        .def("updateMatrices", &Projection3D::updateMatrices)
        .def("setPose", &Projection3D::setPose)
        .def_readonly("type", &Projection3D::type)
        .def_readwrite("det_count_u", &Projection3D::det_count_u)
        .def_readwrite("det_spacing_u", &Projection3D::det_spacing_u)
        .def_readwrite("det_count_v", &Projection3D::det_count_v)
        .def_readwrite("det_spacing_v", &Projection3D::det_spacing_v)
        .def_readwrite("s_dist", &Projection3D::s_dist)
        .def_readwrite("d_dist", &Projection3D::d_dist)
        .def_readwrite("pitch", &Projection3D::pitch)
        .def_readwrite("n_angles", &Projection3D::n_angles)
        .def_property(
            "imageToWorld",
            [](const Projection3D &proj)
            { return poseToNumpy(proj.imageToWorld); },
            [](Projection3D &proj, py::array_t<float> &m)
            { return numpyToPose(proj.imageToWorld, m); })
        .def_property_readonly("worldToImage", [](const Projection3D &proj)
                               { return poseToNumpy(proj.worldToImage); })
        // .def_property_readonly("voxelToWorld", [](const Projection3D& proj) { return poseToNumpy(proj.voxelToWorld); })
        .def_property_readonly("worldToVoxel", [](const Projection3D &proj)
                               { return poseToNumpy(proj.worldToVoxel); });

    py::class_<ExecCfg>(m, "ExecCfg").def(py::init<int, int, int, int>());

    py::class_<SymbolicFunction>(m, "SymbolicFunction")
        .def(py::init<float, float>())
        .def("add_gaussian", &SymbolicFunction::add_gaussian)
        .def("add_ellipse", &SymbolicFunction::add_ellipse)
        .def("move", &SymbolicFunction::move)
        .def("scale", &SymbolicFunction::scale);
}