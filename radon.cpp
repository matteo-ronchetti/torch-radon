#include <torch/extension.h>

#include <iostream>
#include <vector>

void radon_forward(torch::Tensor x) {
  std::cout << x.data<float>()[0] << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &radon_forward, "Radon forward");
}
