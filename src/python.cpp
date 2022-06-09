#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Tensor.h"

namespace py = pybind11;

Tensor tensorFromBuffer(const py::buffer &array, float spacing_x = 1.0, float spacing_y = 1.0, float spacing_z = 1.0)
{
    py::buffer_info info = array.request();
    std::vector<int> shape;
    for (int i = 0; i < info.ndim; i++)
        shape.push_back(info.shape[i]);

    auto type = Tensor::Type::Float;
    return Tensor(shape, type, info.ptr, false, -1);
}

PYBIND11_MODULE(tr, m)
{
    py::enum_<Tensor::Type>(m, "Type")
        .value("Float", Tensor::Type::Float)
        .value("Half", Tensor::Type::Half)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def(py::init(&tensorFromBuffer))
        .def("isCuda", &Tensor::isCuda)
        .def("device", &Tensor::device)
        .def("cuda", &Tensor::cuda)
        .def("cpu", &Tensor::cpu)
        .def("shape", &Tensor::shape)
        .def("dimensions", &Tensor::dimensions)
        .def("byteSize", &Tensor::byteSize)
        .def("isOwning", &Tensor::isOwning);
}