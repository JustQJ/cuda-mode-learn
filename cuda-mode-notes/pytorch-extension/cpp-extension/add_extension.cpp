#include <torch/extension.h>

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
  return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add two tensors");
}