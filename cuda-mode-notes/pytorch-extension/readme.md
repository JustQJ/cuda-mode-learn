# torch.utils.cpp_extension 
1. 作用：用于在Pytroch中添加C++拓展模块
2. 方式：通常有三种常用的方式：（1）使用`setuptools`工具对`cpp`和`cu`源文件进行模块的构建，是提前构建好然后在python代码中加载；（2）使用`torch.utils.cpp_extension.load()`从源文件进行加载模块，但是直接在python运行中进行及时编译，而不是提前编译好；（3）使用`torch.utils.cpp_extension.load_inline()`直接从字符串中编译和加载拓展模块，不需要额外的源文件。

## setuptools工具构建模块
`torch.utils.cpp_extension`中有三个主要的类，`CppExtension`、`CUDAExtension`和`BuildExtension`。                
- `CppExtension` 主要用于构建C++拓展模块；
- `CUDAExtension`继承自`CppExtension`，用于构建包括CUDA代码的拓展模块；
- `BuildExtension`是`setuptools`的命令拓展，用于构建拓展模块。`CppExtension`和`CUDAExtension`都需要通过`BuildExtension`进行构建。

1. `CppExtension`示例     
    创建`add_extension.cpp`文件         
    ```c++
    #include <torch/extension.h>

    torch::Tensor add(torch::Tensor a, torch::Tensor b) {
        return a + b;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add", &add, "A function that adds two tensors");
    }
    ```
    创建`setup.py`文件
    ```python
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    setup(
        name='add_extension', ## 包的名字，主要用于分发和pip安装包
        ext_modules=[
            CppExtension(
                name='add_extension', 
                sources=['add_extension.cpp'],
                extra_compile_args=['-g'],
                ), ## 第一个是模块的名字，用于在python中导入使用，不一定和包名相同
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    ```
    使用命令`python setup.py build_ext --inplace`进行构建，其中`--inplace`表示将构建的拓展模块放在源码目录中，方便python直接在当前目录导入。                 

    在python中调用模块
    ```python

    import add_extension

    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = add_extension.add(a, b)

    print(c)  # Output should be tensor([5, 7, 9])


    ```

2. `CUDAExtension`示例   
    创建`add_extension.cu`文件
    ```cuda
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>

    __global__ void add_kernel(float* x, float* y, float* out, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            out[idx] = x[idx] + y[idx];
        }
    }

    torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) { //定义cpp文件中的函数
        auto out = torch::zeros_like(x);
        int size = x.numel();
        float *x_data = x.data_ptr<float>();
        float *y_data = y.data_ptr<float>();
        float *out_data = out.data_ptr<float>();

        int threads = 1024;
        int blocks = (size + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(x_data, y_data, out_data, size);

        return out;
    }

    ```
    创建`add_extension.cpp`文件         
    ```c++
    #include <torch/extension.h>

    torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y); //声明cu文件中的函数

    torch::Tensor add(torch::Tensor x, torch::Tensor y) { //用于外部调用
        return add_cuda(x, y);
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add", &add, "Add two tensors using CUDA");
    }

    ```
    创建`setup.py`文件
    ```python
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    setup(
        name='add_cuda_extension', ## 包的名字，主要用于分发和pip安装包
        ext_modules=[
            CUDAExtension( ## cuda的拓展
                name='add_cuda_extension', 
                sources=['add_extension.cpp', 'add_extension.cu']
                extra_compile_args={'cxx':['-g'],
                                    'nvcc':['-O2']},## 选择编译器和编译选项
                ), ## 第一个是模块的名字，用于在python中导入使用，不一定和包名相同
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
    ```
    使用命令`python setup.py build_ext --inplace`进行构建                    

    在python中调用模块
    ```python

    import add_cuda_extension

    a = torch.tensor([1, 2, 3]).cuda()
    b = torch.tensor([4, 5, 6]).cuda()
    c = add_cuda_extension.add(a, b)

    print(c) 
    ```
3. 使用pip安装，使得整个python环境都可以调用
    **TODO**

## `load()`函数构建模块
同样需要源文件                

创建`add_extension.cu`文件
```cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_kernel(float* x, float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] + y[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) { //定义cpp文件中的函数
    auto out = torch::zeros_like(x);
    int size = x.numel();
    float *x_data = x.data_ptr<float>();
    float *y_data = y.data_ptr<float>();
    float *out_data = out.data_ptr<float>();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(x_data, y_data, out_data, size);

    return out;
}

```
创建`add_extension.cpp`文件         
```c++
#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y); //声明cu文件中的函数

torch::Tensor add(torch::Tensor x, torch::Tensor y) { //用于外部调用
    return add_cuda(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors using CUDA");
}

```
直接在python文件中构建，而不是使用`set_up.py`文件。               
```python
import torch
from torch.utils.cpp_extension import load

# 加载 CUDA 扩展
add_extension = load(
    name='add_cuda_extension', 
    sources=['add_extension.cpp', 'add_extension.cu']
    extra_cflags=['-O2'],
    verbose=True,
    ## build_directory = './temp' ##该参数会放置编译的文件的
    )

# 使用扩展模块
a = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
b = torch.tensor([4, 5, 6], dtype=torch.float32).cuda()
print(add_extension.add(a, b))  # 输出 tensor([5, 7, 9], device='cuda:0')

```
    

## `load_inline()`函数构建模块
不需要源文件，直接把代码字符写在python文件中
```python
import torch
from torch.utils.cpp_extension import load_inline

## C++ 代码字符串
cpp_source = """
#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "A function that adds two tensors");
}
"""

# 加载 C++ 扩展
add_extension = load_inline(name='inline_extension',
                     cpp_sources=cpp_source,
                     extra_cflags=['-O3'])

# 使用扩展模块
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(add_extension.add(a, b))  # 输出 tensor([5, 7, 9])

```


# 参考链接
https://pytorch.org/docs/stable/cpp_extension.html                     
https://www.cntofu.com/book/169/docs/1.0/docs_cpp_extension.md                        
https://github.com/cuda-mode/lectures                       


