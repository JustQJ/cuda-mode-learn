#include <torch/extension.h>
torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y);

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
  return add_cuda(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add two tensors with cuda kernel");
}
