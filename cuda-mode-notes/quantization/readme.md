

# 基本知识
## 量化原理
将浮点矩阵量化到$k$位的int类型，就是做一个映射函数，即
$$
f(x) = x/s - z
$$
其中$x$ 是一个张量，$s$ 通过 $x$ 的最大最小值和$k$位int的最大最小值进行计算，即
$$
s = (max(x)- min(x)) / (2^k-1)\\
z = round(max(x)/s) - (2^k-1) \\
q_x = clip(round(x/s-z), 0, 2^k-1)
$$
这里是将$x$投射到$[0, 2^k-1]$，当然也可以是$[-2^{(k-1)}, 2^{(k-1)}-1]$，但是全是正数，实际的实现会更好。              
也可以使用对称映射，即将$[-m, m]$映射到$[-2^{(k-1)}+1, 2^{(k-1)}-1]$，有
$$
z = 0 \\
s = m/(2^{(k-1)}-1) \\
m = max(abs(max(x)), abs(min(x)))
$$

最大最小值映射是最基本的，现在的方法主要的优化基本就是优化离群值，即$x$中有一些值异常的大或者小，分布不均匀。         

## 量化粒度

**per-tensor quantization:** 一个张量矩阵使用同一个scale和zero，scale和zero占用的内存小，当然误差变大。

**per-channel quantization:** 一个通道使用一个scale和zero（对于一个二维权重矩阵，就是一行或者一列），scale和zero占用的内存大，当然误差变小。

**per-group quantization:** 对上面进行折中，将多个通道形成一个group，然后一个group使用一个scale和zero，例如一般使用128作为group的大小。也可以是多个元素形成一个group，即比per-channel粒度更小。


## 量化种类
1. 只量化权重：在计算是需要先将权重反量化为激活的类型，然后计算。
2. 权重+激活量化：在计算时不需要反量化，而是将激活也量化为权重的类型，然后进行计算。


**量化的主要好处：（1）减少模型的权重的内存占用；（2）减少计算过程中对内存带宽的需求。**


# 实际实现
一般通过以下几步实现：
1. 使用量化函数对输入的tensor进行量化，将每个元素量化到int k的范围内。
    ```python
    def quantize(x: torch.Tensor, bit: int, group_size: int) -> torch.Tensor:
        """
        x: input tensor
        bit: quantization bit
        group_size: group size for quantization, -1 means per tensor quantization, group_size menas per group quantization, each group has group_size rows
        """

        group_num = (x.shape[0] + group_size-1) // group_size if group_size > 0 else 1 
        zeros = torch.zeros(group_num, dtype=x.dtype, device=x.device) ## zero for each group
        scales = torch.zeros(group_num, dtype=x.dtype, device=x.device) ## scale for each group
        group_idxs = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device) ## group index for each row
        qw = torch.zeros_like(x)
        for i in range(group_num): ## each group use same s and z
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

        qw = qw.to(torch.int32) ##
        return qw, scales, zeros, group_idxs
    
    
    ```

    该函数返回的量化权重还是每个元素使用int32存储，内存占用并没有变小，但是全部在intk的范围内。

2. 对量化的权重进行打包，即使用一个32位的int存储多个intk的值，就是将$32/k$行的数据压缩到一行存储。
    ```python
    def pack_matrix(x: torch.Tensor, bit: int) -> torch.Tensor:
        assert x.dtype == torch.int32
        device = x.device
        np_x = x.cpu().numpy().astype(np.uint32) 
        in_features = x.shape[0]
        out_features = x.shape[1]
        n_number_per_int = 32 // bit 
        qweight = np.zeros(
            (in_features// n_number_per_int, out_features), dtype=np.uint32
        ) ## 

        max_q = 2**bit - 1

        for row in range(qweight.shape[0]):
            for j in range(n_number_per_int):
                qweight[row] |= (np_x[row*n_number_per_int + j] ) << (bit * j) ## main operation

        qweight = qweight.astype(np.int32)

        return torch.from_numpy(qweight).to(device)
    ```
    重点是下面这行代码
    ```
    qweight[row] |= (np_x[row*n_number_per_int + j]) << (bit * j)
    ```
    即将`x[i][0], x[i+1][0],...,x[i+32/k-1][0]` 一共$32/k$个元素放在了一起，分别存储在32位的不同位置。               
    例如对于$k=8$, 则`x[i][0]`这个32位的整数在$0-255$之间，即其只有最后八位是有用的，其他高24位均为0，所以通过移位和或操作，将每个8位放在对应的位置。具体来说就是`x[i][0]`放在0-7位，`x[i+1][0]`放在8-15位，`x[i+2][0]`放在16-23位，`x[i+3][0]`放在24-31位。            
    因此将m行变成了$m/(32/k)$行。  


3. 实现相应的kernel函数，即在计算过程中需要对打包的数据进行解压缩。

    ```python
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



    def w248a16_matmul(
        w:torch.Tensor, 
        a:torch.Tensor, 
        scales: torch.Tensor, 
        zeros: torch.Tensor, 
        group_idxs: torch.Tensor, 
        bit: int
        ):
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
    ```
    以上代码使用`triton`实现，相比于`cuda`方便在python中使用。  
    相比于普通的矩阵乘法乘法，主要注意的是kernel的实现过程中，需要搞清楚权重矩阵的读取、解压缩、反量化。
    ```
    w_ptrs = w_ptr + (offset_wm[:, None]//n_per_int * stride_wm + offset_k[None, :]*stride_wk)
    ```           
    权重矩阵的地址，每连续的$32/k$行的读取的地址是相同的，所以除以$32/k$。

    ```
    shifter = (offset_wm%n_per_int)*bit
    w = (w >> shifter[:, None]) & max_q
    ```
    需要对每个元素进行解码，即通过位移和与的操作，把每个段里的元素单独解码出来，即把一个32位的数还原成$32/k$个数。
    
    ```
    w = (w+zeros[: , None])*scales[:, None]
    ```
    然后对每个元素进行反量化。

**上面的实现很简单，没有优化，所以误差很大，同时速度也比较慢。**






# 参考
https://iq.opengenus.org/basics-of-quantization-in-ml/

https://huggingface.co/docs/optimum/concept_guides/quantization

https://github.com/fpgaminer/GPTQ-triton/blob/main/quantize.py

https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_tritonv2.py