from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='add_extension', # name of the package
    ext_modules=[
        CppExtension(
            name='add_extension', # name of module, imported by python script
            sources=['add_extension.cpp'],
            extra_compile_args=['-g'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)