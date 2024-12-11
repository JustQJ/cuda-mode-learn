import math
import time

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse, ## for backprop, current not compute in this kernel
    softmax_scale,
    stride_qb, ## batch
    stride_qh, ## head number
    stride_qm, ## Q seq length
    stride_kb, 
    stride_kh,
    stride_kn, ## K,V Cache seq length
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    headdim,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    BLOCK_M: tl.constexpr, ## seq length dim block size
    BLOCK_N: tl.constexpr, ## batch * head dim block size
    BLOCK_HEADDIM: tl.constexpr, ## head dim block size the first power of 2 that >= headdim
):
    start_m = tl.program_id(0) ## seq length dim id 
    off_hb = tl.program_id(1) ## batch and head dim id, these two dim merged into one
    off_b = off_hb // nheads ## batch dim id
    off_h = off_hb % nheads ## head dim id of the current batch

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) ## this block ids for seq dim

    offs_n = tl.arange(0, BLOCK_N) ## this block ids for k,v seq dim, each iteration will process a block of kv seq dim, need traverse all kv seq dim
    offs_d = tl.arange(0, BLOCK_HEADDIM) ## this block ids for head dim

    ## offs_m[:, None] will add one dim in the end, shape[128] -> shape[128, 1], while offs_d[None, :] will add one dim in the front, shape[128] -> shape[1, 128]
    ## offs_m[:, None]*stride_qm + offs_d[None, :] will broadcast, the shape is [128, 128]
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None]*stride_qm + offs_d[None, :] 
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None]*stride_kn + offs_d[None, :]
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None]*stride_vn + offs_d[None, :]
    )

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) ## record each row's 
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") ## record each row's max value
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32) ## record output of this block and udpate each iterator

    ## we suppose the head_dim is equal to BLOCK_HEADDIM, so we can load q directly
    q = tl.load(q_ptrs, mask=offs_m[:, None]<seqlen_q, other=0.0) ## load q, mask all length dim > seqlen_q

    end_n = seqlen_k ## all kv-cache length

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        ## compute q,k
        ## k_ptrs+start_n*stride_kn, each time load a block of kv-cache, BLOCK_N rows
        ## start_n*stride_kn is the start row 
        ## compute current block 
        k = tl.load(k_ptrs+start_n*stride_kn, mask=(start_n+offs_n[:,None])<seqlen_k, other=0.0)
        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k))

        ## mask all paddings that larger the seqlen with -inf
        qk += tl.where((start_n+offs_n)[None, :]<seqlen_k, 0, float("-inf"))

        ## new m(x) value of each row
        m_new = tl.maximum(tl.max(qk, 1)*softmax_scale, m_i)
        p = tl.exp(qk*softmax_scale - m_new[:, None])
        l_ij = tl.sum(p, 1) # l(x) for current block

        

        acc_o_scale = tl.exp(m_i - m_new) 

        acc_o = acc_o * acc_o_scale[:, None] ## previous o been recorrected

        v = tl.load(v_ptrs+start_n*stride_vn, mask=(start_n+offs_n[:,None])<seqlen_k, other=0.0)

        p = p.to(v.dtype)
        acc_o = tl.dot(p, v, acc_o)

        m_i = m_new ## update m(x)
        l_i = acc_o_scale*l_i + l_ij ## update l(x)

    
    acc_o = acc_o / l_i[:, None] ## compute the final output by l(x)

    output_ptrs = (
        Out + off_b*stride_ob + off_h*stride_oh + offs_m[:, None]*stride_om + offs_d[None, :]
    )

    acc_o = acc_o.to(Out.type.element_ty)

    tl.store(output_ptrs, acc_o, mask=offs_m[:, None]<seqlen_q)

    


def flash_attn_forward(q, k, v, softmax_scale=None):
    # shape constraints
    batch, nheads, seqlen_q, d = q.shape
    _, nheads, seqlen_k, _ = k.shape
    assert k.shape == (batch, nheads, seqlen_k, d)
    assert v.shape == (batch, nheads, seqlen_k, d)
    assert d in [16, 32, 64, 128], "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    # tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = d
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        nheads,
        d,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        num_warps=num_warps,
        num_stages=1,
    )

    return o
    


def basic_attention(q, k, v, softmax_scale=None):
    batch, nheads, seqlen_q, d = q.shape
    _, nheads, seqlen_k, _ = k.shape
    assert k.shape == (batch, nheads, seqlen_k, d)
    assert v.shape == (batch, nheads, seqlen_k, d)
    assert d in [16, 32, 64, 128], "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    attention = torch.nn.functional.softmax(scores, dim=-1)

    output = torch.matmul(attention, v)

    return output


def test():
    batch = 16
    nheads = 32
    seqlen_q = 128
    seqlen_k = 128
    d = 128
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(batch, nheads, seqlen_q, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, nheads, seqlen_k, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, nheads, seqlen_k, d, device="cuda", dtype=torch.bfloat16)
    output_f = flash_attn_forward(q, k, v, softmax_scale=scale)
    output_b = basic_attention(q, k, v, softmax_scale=scale)

    # correct = torch.allclose(output_f, output_b, atol=1e-2)

    relative_error = torch.norm(output_f - output_b) / torch.norm(output_b)
    

    print(f"flash attention otuput: {output_f[0][0]}")
    print(f"basic attention output: {output_b[0][0]}")
    print(f"relative error: {relative_error}")


def bench_speed():
    bss = [1, 4,8,16,32]
    nheads = 32
    seqlens = [(1, 16), (1, 128), (1, 256), (1, 512), (16, 16), (128, 128), (512, 512), (1024, 1024)]
    d = 128
    scale = 1.0 / math.sqrt(d)
    iter_num = 100
    for bs in bss:
        for (sq_q, sq_k) in seqlens:
            q = torch.randn(bs, nheads, sq_q, d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(bs, nheads, sq_k, d, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(bs, nheads, sq_k, d, device="cuda", dtype=torch.bfloat16)

            ## wwarm up
            for _ in range(10):
                output_f = flash_attn_forward(q, k, v, softmax_scale=scale)

            torch.cuda.synchronize()
            t1 = time.time()
            for _ in range(iter_num):
                output_f = flash_attn_forward(q, k, v, softmax_scale=scale)

            torch.cuda.synchronize()
            t2 = time.time()
            flash_time = (t2 - t1) / iter_num


            ## wwarm up
            for _ in range(10):
                output_b = basic_attention(q, k, v, softmax_scale=scale)
            
            torch.cuda.synchronize()
            t1 = time.time()
            for _ in range(iter_num):
                output_b = basic_attention(q, k, v, softmax_scale=scale)
            torch.cuda.synchronize()
            t2 = time.time()

            basic_time = (t2 - t1) / iter_num

            print(f"bs: {bs}, (sq_q, sq_k): {(sq_q, sq_k)}, flash_time: {flash_time}, basic_time: {basic_time}, speedup: {basic_time/flash_time}")
           



if __name__ == "__main__":
    test()
    bench_speed()



    