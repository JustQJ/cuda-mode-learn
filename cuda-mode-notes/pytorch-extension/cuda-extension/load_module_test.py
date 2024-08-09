import os
import torch
from torch.utils.cpp_extension import load

build_directory = './load_build'
if not os.path.exists(build_directory):
    os.makedirs(build_directory)

add_extension = load(
    name='add_cuda_extension',
    sources=[
        'add_extension_cuda.cpp',
        'add_extension_cuda.cu'
    ],
    extra_cflags=['-O2'],
    verbose=True,
    build_directory=build_directory
)

a = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
b = torch.tensor([4, 5, 6], dtype=torch.float32).cuda()
c = add_extension.add(a, b)
print(c)