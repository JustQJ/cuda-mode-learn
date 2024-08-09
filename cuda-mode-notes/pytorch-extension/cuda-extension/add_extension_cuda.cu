#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_kernel(float *x, float *y, float *out, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < size){
        out[idx] = x[idx] + y[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y){
    auto out = torch::empty_like(x);
    int size = x.numel();

    add_kernel<<<(size+256-1)/256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}