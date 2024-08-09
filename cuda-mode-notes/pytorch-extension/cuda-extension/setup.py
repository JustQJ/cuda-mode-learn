from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='add_cuda_extension', # name of the package
    ext_modules=[
        CUDAExtension(
            name='add_cuda_extension', # name of module, imported by python script
            sources=['add_extension_cuda.cpp', 'add_extension_cuda.cu'],
            # extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']},
            # extra_link_flags=['-Wl,--no-as-needed', '-lcuda']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)