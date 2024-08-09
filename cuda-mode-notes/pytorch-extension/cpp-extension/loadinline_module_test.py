"""
test the module build with load_inline
"""


import os
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """

#include <torch/extension.h>
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return a + b;
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors");
}

"""

build_directory = './loadinline_build'
if not os.path.exists(build_directory):
    os.makedirs(build_directory)

add_extension = load_inline(
    name='add_extension',
    cpp_sources=cpp_source,
    # functions=['add'], ## when don't use PYBIND11_MODULE, we should set the functions
    extra_cflags=['-O2'],
    build_directory=build_directory
    
)
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

c = add_extension.add(a, b)
print(c)