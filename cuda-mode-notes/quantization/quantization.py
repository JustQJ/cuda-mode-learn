import time
import torch
import triton
import triton.language as tl
import numpy as np

"""
# quantize function
we don't use the negative value, so the range is [0, 2^bit-1]

q_x = x/scale - zero # 方便反量化的计算

x = (q_x + zero) * scale

f(x) = clip(round(x/scale - zero), 0, 2^bit-1)

x: input tensor
scale: scaling factor
zero: zero point
output: quantized tensor

scale =  (max(x) - min(x)) / (2^bit - 1)
zero =  round(max(x)/scale) - (2^bit - 1)



## if use symmetric quantization
## [-A1, A1] -> [-(2^(bit-1)-1), 2^(bit-1)-1]
scale = (2^(bit-1)-1) / max(|min(x)|, max(x)) 
zero = 0
"""


def quantize(x: torch.Tensor, bit: int, group_size: int) -> torch.Tensor:
    """
    x: input tensor
    bit: quantization bit
    group_size: group size for quantization, -1 means per tensor quantization, group_size menas per group quantization, each group has group_size rows
    """
    # max_val = x.max()
    # min_val = x.min()
    # if min_val < 0:
    #     scale = (2 ** bit - 1) / max(abs(min_val), max_val)
    #     zero = 2 ** (bit - 1) - 1 - round(scale * max_val)
    # else:
    #     scale = (2 ** (bit - 1) - 1) / max_val
    #     zero = 0

    ## using asymmetric quantization
    ## quantize to [0, 2^bit-1]
    group_num = (x.shape[0] + group_size-1) // group_size if group_size > 0 else 1 
    zeros = torch.zeros(group_num, dtype=x.dtype, device=x.device) ## zero for each group
    scales = torch.zeros(group_num, dtype=x.dtype, device=x.device) ## scale for each group
    group_idxs = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device) ## group index for each row
    qw = torch.zeros_like(x)
    for i in range(group_num):
        start = i * group_size
        end = min((i + 1) * group_size, x.shape[0])
        group_x = x[start:end]
        
        max_val = group_x.max()
        min_val = group_x.min()
        scale =  (max_val - min_val) /(2 ** bit - 1)
        zero = torch.round(max_val/scale) - (2 ** bit - 1)
        scales[i] = scale
        zeros[i] = zero

        group_idxs[start:end] = i

        qw[start:end] = torch.clip(torch.round(group_x/scale - zero), 0, 2**bit - 1)

    qw = qw.to(torch.int32)
    return qw, scales, zeros, group_idxs



def dequantize(x, scales, zeros, group_idxs, data_typ=torch.float32) -> torch.Tensor:
    # if x.dtype == torch.int8:
    #     x1 = x.to(torch.float32)

    x1 = x.to(data_typ)
    for i in range(x.shape[0]):
        scale = scales[group_idxs[i]]
        zero = zeros[group_idxs[i]]
        x1[i] = (x1[i]+zero) * scale

    return x1
    
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]
@triton.autotune( ## 首次调用时会搜索上面的配置找到最优的，然后后面调用的时候就不会再搜索了，依据是key是否改变
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)

## quantize matrix matmul
## W8A32, W is int8, A is float32
## W: [M, K], A: [K, N], C: [M, N]

@triton.jit
def w8a32_matmul_kernel(
    w_ptr, a_ptr, res_ptr,  ## w [M//(32/bit), K ] is int32, a [K, N] is float32, res [M, K] is float32
    M, N, K,
    scales_ptr, zeros_ptr, group_idxs_ptr,
    bit,
    max_q,

    stride_wm, stride_wk, ## 表示每个维度的地址偏移
    stride_ak, stride_an,
    stride_rm, stride_rn,

    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    n_per_int = 32 // bit

    offset_wm = (pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
    offset_an = (pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    

    ## repeat the offset_wm for n_per_int times
    ##就是每n_per_int个行的访问的地址是一样的
    w_ptrs = w_ptr + (offset_wm[:, None]//n_per_int * stride_wm + offset_k[None, :]*stride_wk)
    a_ptrs = a_ptr + (offset_k[:, None] * stride_ak + offset_an[None, :]*stride_an)
    
    ## shift for w data
    shifter = (offset_wm%n_per_int)*bit ## 0, 8, 16, 24
    
    

    accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    group_idx_ptrs = group_idxs_ptr + offset_wm
    
    group_idxs = tl.load(group_idx_ptrs)

    scale_ptrs = scales_ptr + group_idxs
    zero_ptrs = zeros_ptr + group_idxs
    zeros = tl.load(zero_ptrs)
    scales = tl.load(scale_ptrs)
    #https://github.com/triton-lang/triton/issues/4652
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), 1, num_stages=1):
        w = tl.load(w_ptrs, mask=offset_k[None, :] < K-k*BLOCK_SIZE_K, other=0)
        a = tl.load(a_ptrs, mask=offset_k[:, None] < K-k*BLOCK_SIZE_K, other=0)

        # print(w)

        ## dequantize w
        # w = w.to(tl.float32)
        # w = (w - zero)/scale

        ## unpack w
        w = (w >> shifter[:, None]) & max_q
        ## dequantize w
        w = (w+zeros[: , None])*scales[:, None]
        

        accu = tl.dot(w, a, accu)

        w_ptrs += BLOCK_SIZE_K * stride_wk
        a_ptrs += BLOCK_SIZE_K * stride_ak
    

    offset_rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    res_ptrs = res_ptr + (offset_rm[:, None] * stride_rm + offset_rn[None, :]*stride_rn)

    c_mask = (offset_rm[:, None] < M) & (offset_rn[None, :] < N)    

    tl.store(res_ptrs, accu, mask=c_mask)



def w8a32_matmul(w:torch.Tensor, a:torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_idxs: torch.Tensor):

    assert w.dtype == torch.int32
    assert a.dtype == torch.float32
    assert w.is_cuda and a.is_cuda
    assert w.shape[1] == a.shape[0]
    
    bit = 8
    max_q = 2**bit - 1
    m, K = w.shape
    K, N = a.shape
    M = m * (32//bit)
    res = torch.zeros(M, N, dtype=torch.float32, device=w.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),) 
    w8a32_matmul_kernel[grid](w, a, res, M, N, K,scales , zeros, group_idxs, bit,max_q, w.stride(0), w.stride(1), a.stride(0), a.stride(1), res.stride(0), res.stride(1))

    return res 



# @triton.autotune( ## 首次调用时会搜索上面的配置找到最优的，然后后面调用的时候就不会再搜索了，依据是key是否改变
#     configs=get_cuda_autotune_config(),
#     key=['M', 'N', 'K'],
# )

# @triton.jit
# def w8a8_matmul_kernel(
#     w_ptr, a_ptr, res_ptr,  ## w is int8, a is float32, res is float32
#     M, N, K,
#     scale, zero,
#     scale_a, zero_a,

#     stride_wm, stride_wk, ## 表示每个维度的地址偏移
#     stride_ak, stride_an,
#     stride_rm, stride_rn,

#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr
# ):
#     pid = tl.program_id(0)
    
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     pid_m = pid // num_pid_n
#     pid_n = pid % num_pid_n

#     offset_wm = (pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
#     offset_an = (pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
#     offset_k = tl.arange(0, BLOCK_SIZE_K)

#     w_ptrs = w_ptr + (offset_wm[:, None] * stride_wm + offset_k[None, :]*stride_wk)
#     a_ptrs = a_ptr + (offset_k[:, None] * stride_ak + offset_an[None, :]*stride_an)

    
    

#     accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     # z = tl.load(zero)
#     # s = tl.load(scale)
#     #https://github.com/triton-lang/triton/issues/4652
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         w = tl.load(w_ptrs, mask=offset_k[None, :] < K-k*BLOCK_SIZE_K, other=0)
#         a = tl.load(a_ptrs, mask=offset_k[:, None] < K-k*BLOCK_SIZE_K, other=0)

#         # print(w)

#         # ## dequantize w
#         # w = w.to(tl.float32)
#         w = (w - zero)/scale

#         # ## dequantize a
#         # a = a.to(tl.float32)
#         a = (a - zero_a)/scale_a

        

#         accu = tl.dot(w, a, accu)

#         w_ptrs += BLOCK_SIZE_K * stride_wk
#         a_ptrs += BLOCK_SIZE_K * stride_ak
    

#     offset_rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offset_rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

#     res_ptrs = res_ptr + (offset_rm[:, None] * stride_rm + offset_rn[None, :]*stride_rn)

#     c_mask = (offset_rm[:, None] < M) & (offset_rn[None, :] < N)    
#     accu = accu.to(tl.float32)
#     tl.store(res_ptrs, accu, mask=c_mask)


# def w8a8_matmul(w:torch.Tensor, a:torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, scale_a: torch.Tensor, zero_a: torch.Tensor):

#     assert w.dtype == torch.int8
#     assert a.dtype == torch.int8
#     assert w.is_cuda and a.is_cuda
#     assert w.shape[1] == a.shape[0]

#     M, K = w.shape
#     K, N = a.shape

#     res = torch.zeros(M, N, dtype=torch.float32, device=w.device)
#     grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),) 
#     w8a8_matmul_kernel[grid](w, a, res, M, N, K,scale.item(), zero.item(),scale_a.item(), zero_a.item(), w.stride(0), w.stride(1), a.stride(0), a.stride(1), res.stride(0), res.stride(1))

#     return res 



## WkA16, W is intk, A is float16, k in [2,4,8]
@triton.autotune( ## 首次调用时会搜索上面的配置找到最优的，然后后面调用的时候就不会再搜索了，依据是key是否改变
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)

@triton.jit
def w248a16_matmul_kernel(
    w_ptr, a_ptr, res_ptr,  ## w [M//(32/bit), K ] is int32, a [K, N] is float16, res [M, K] is float16
    M, N, K,
    scales_ptr, zeros_ptr, group_idxs_ptr,
    bit,
    max_q,

    stride_wm, stride_wk, ## 表示每个维度的地址偏移
    stride_ak, stride_an,
    stride_rm, stride_rn,

    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr ## group load for optimizing the L2 cache, we don't use it here
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    n_per_int = 32 // bit

    offset_wm = (pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
    offset_an = (pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
    offset_k = tl.arange(0, BLOCK_SIZE_K)

    

    ## repeat the offset_wm for n_per_int times
    ##就是每n_per_int个行的访问的地址是一样的
    w_ptrs = w_ptr + (offset_wm[:, None]//n_per_int * stride_wm + offset_k[None, :]*stride_wk)
    a_ptrs = a_ptr + (offset_k[:, None] * stride_ak + offset_an[None, :]*stride_an)
    
    ## shift for w data
    shifter = (offset_wm%n_per_int)*bit ## the shift for each row
    
    

    accu = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    group_idx_ptrs = group_idxs_ptr + offset_wm
    
    group_idxs = tl.load(group_idx_ptrs)

    scale_ptrs = scales_ptr + group_idxs
    zero_ptrs = zeros_ptr + group_idxs
    zeros = tl.load(zero_ptrs)
    scales = tl.load(scale_ptrs)
    #https://github.com/triton-lang/triton/issues/4652
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), 1, num_stages=1):
        w = tl.load(w_ptrs, mask=offset_k[None, :] < K-k*BLOCK_SIZE_K, other=0)
        a = tl.load(a_ptrs, mask=offset_k[:, None] < K-k*BLOCK_SIZE_K, other=0)

        # print(w)

        ## dequantize w
        # w = w.to(tl.float32)
        # w = (w - zero)/scale

        ## unpack w
        w = (w >> shifter[:, None]) & max_q
        ## dequantize w
        w = (w+zeros[: , None])*scales[:, None]
        

        accu = tl.dot(w, a, accu)

        w_ptrs += BLOCK_SIZE_K * stride_wk
        a_ptrs += BLOCK_SIZE_K * stride_ak
    

    offset_rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    res_ptrs = res_ptr + (offset_rm[:, None] * stride_rm + offset_rn[None, :]*stride_rn)

    c_mask = (offset_rm[:, None] < M) & (offset_rn[None, :] < N)    
    accu = accu.to(tl.float16)
    tl.store(res_ptrs, accu, mask=c_mask)



def w248a16_matmul(w:torch.Tensor, a:torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_idxs: torch.Tensor, bit: int):
    assert w.dtype == torch.int32
    assert a.dtype == torch.float16
    assert w.is_cuda and a.is_cuda
    assert w.shape[1] == a.shape[0]
    assert bit in [2, 4, 8]
    
    
    max_q = 2**bit - 1
    m, K = w.shape
    K, N = a.shape
    M = m * (32//bit)
    res = torch.zeros(M, N, dtype=torch.float16, device=w.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),) 
    w248a16_matmul_kernel[grid](w, a, res, M, N, K,scales, zeros, group_idxs, bit,max_q, w.stride(0), w.stride(1), a.stride(0), a.stride(1), res.stride(0), res.stride(1))

    return res 





def pack_matrix(x: torch.Tensor, bit: int) -> torch.Tensor:
    assert x.dtype == torch.int32
    device = x.device
    np_x = x.cpu().numpy().astype(np.uint32) 
    in_features = x.shape[0]
    out_features = x.shape[1]
    n_number_per_int = 32 // bit
    qweight = np.zeros(
        (in_features// n_number_per_int, out_features), dtype=np.uint32
    ) ## shape is [128//32*8, 256], 4 rows are packed into 1 row

    max_q = 2**bit - 1 ## 255, 15

    for row in range(qweight.shape[0]):
        for j in range(n_number_per_int):
            # qweight[row] |= (np_x[row*n_number_per_int + j] & max_q) << (bit * j)
            qweight[row] |= (np_x[row*n_number_per_int + j]) << (bit * j) ##全是正数，高位全是0，不需要&max_q

    qweight = qweight.astype(np.int32)

    return torch.from_numpy(qweight).to(device)


def unpack_matrix(x: torch.Tensor, bit: int) -> torch.Tensor:
    assert x.dtype == torch.int32
    np_x = x.numpy().astype(np.uint32) 
    n_number_per_int = 32 // bit
    in_features = x.shape[0] * n_number_per_int
    out_features = x.shape[1]
    
    d_qweight = np.zeros((in_features, out_features), dtype=np.int32)

    max_q = 2**bit - 1 ## 255, 15

    for row in range(x.shape[0]):
        for j in range(n_number_per_int):
            d_qweight[row * n_number_per_int + j] = (np_x[row] >> (bit * j)) & max_q

    return torch.from_numpy(d_qweight)

def check_error():
    w = torch.randn(1024, 1024).cuda().to(torch.float16)
    a = torch.randn(1024, 1024).cuda().to(torch.float16)
    group_size = 128

    for bit in [2, 4, 8]:
        q_w, scales, zeros, group_idxs = quantize(w, bit, group_size)
        d_qw = dequantize(q_w, scales, zeros, group_idxs, torch.float16)

        

        real_output = torch.matmul(w, a)

    
        torch_output = torch.matmul(d_qw, a)


        pack_qw = pack_matrix(q_w, bit)


        triton_output = w248a16_matmul(pack_qw, a, scales, zeros, group_idxs, bit)
        print("result for bit: ", bit)
        print(f"relative error between torch and triton implementation for quantization: {(torch_output-triton_output).abs().max()/ torch_output.abs().max()}")
        print(f"relative error between triton and real output: {(triton_output-real_output).abs().max()/ real_output.abs().max()}")
        print(torch_output)
        print(triton_output)
        print(real_output)



def benchmark():
    w = torch.randn(4096, 4096).cuda().to(torch.float16)
    a = torch.randn(4096, 4096).cuda().to(torch.float16)
    group_size = 128

    iter_num = 100
    for bit in [2, 4, 8]:
        print("benchmark result for bit: ", bit)
        q_w, scales, zeros, group_idxs = quantize(w, bit, group_size)
        d_qw = dequantize(q_w, scales, zeros, group_idxs, torch.float16)
        pack_qw = pack_matrix(q_w, bit)
        
        ## warm up
        for i in range(10):
            torch_output = torch.matmul(d_qw, a)
        torch.cuda.synchronize()
        t1 = time.time()
        for i in range(iter_num):
            torch_output = torch.matmul(d_qw, a)
        torch.cuda.synchronize()
        t2 = time.time()
        cost_time1 = t2 - t1
        print(f"torch time: {cost_time1/iter_num}")

        ## warm up
        for i in range(10):
            triton_output = w248a16_matmul(pack_qw, a, scales, zeros, group_idxs, bit)
        
        torch.cuda.synchronize()
        t1 = time.time()
        for i in range(iter_num):
            triton_output = w248a16_matmul(pack_qw, a, scales, zeros, group_idxs, bit)
        torch.cuda.synchronize()
        t2 = time.time()
        cost_time2 = t2 - t1
        print(f"triton time: {cost_time2/iter_num}")

        print(f"speedup: {cost_time1/cost_time2}")        
        
  



if __name__ == "__main__":
    check_error()
    benchmark()

