## Attention 基础知识
Attention本质上就是计算一个加权和，即对每个value进行加权求和，得到输出。

### Scaled Dot-product Attention
对于$Q^{m\times d_k}, K^{n\times d_k}, V^{n\times d_v}$，计算输出
$$
O^{m\times d_v} = \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V 
$$
$\text{softmax}(*)$ 部分是一个$m\times n$的矩阵，称为注意力分数矩阵$S^{m\times n}$。$QK^T$其实就是每个$Q$中的每个token的编码向量和$K$中的每个token的编码向量做向量内积，计算相似度。然后通过一个$scale$和$softmax$进行归一化得到注意力分数矩阵。那么$S[i,j]$就表示$i$个token和第$j$个token的相似度相对大小。然后和$V$进行矩阵乘法，其实就是做加权和，第$i$个token的输出就是$V$中的每个token的编码向量和其与$i$的分数进行加权和，即
$$
O[i,:] = \sum_{k=0}^n S[i, k] * V[k, :] 
$$

在主流的LLM Decoder架构中，一般是self-attention，即$Q^{m\times d_k}, K^{n\times d_k}, V^{n\times d_v}$ 来自同一个输入。对于输入$x^{m\times h}$，有

$$
Q^{m\times d_k} = x^{m\times h}\times W_Q^{h, d_k}\\
K^{m\times d_k} = x^{m\times h}\times W_K^{h, d_k}\\
V^{m\times d_v} = x^{m\times h}\times W_V^{h, d_v}
$$
其中 $W_Q^{h, d_k}, W_K^{h, d_k}, W_V^{h, d_v}$表示三个权重矩阵。$m=n$，但是当前引入了KV-Cache后，一般$m$等于1，而$n$表达当前的sequence的长度。 同时，一般有$d_k=d_v=h$。

**Masked Attention:** 此外，在当前的Decoder架构中，由于输出是一个个产生的，即前面的token是看不到后面token的。为了保证训练和推理的一致性，训练阶段需要进行mask，使得第$i$个token不会注意到大于$i$的token，本质上就是相应位置的注意力分数为0，即
$$
S[i,j]=0, i<j
$$
本质上就是$S$是一个下三角矩阵（注意训练阶段$m=n$，即$S^{n\times n}$）。在实现过程中，一般是将$\frac{QK^T}{\sqrt{d_k}}$结果的上三角填充$-\inf$，在经过$softmax$后就变成了0。


### Multi-head Attention

使用多个head捕获不同的信息，使得效果更好。在实现上，就是将多个head的结果进行拼接，得到最后的结果，即
$$
O^{m\times d_{v_i}} = \text{Attention}(Q_i, K_i, V_i)\\
O^{m\times d_v} = \text{Concat}(O^{m\times d_{v_0}}, O^{m\times d_{v_1}}, ..., O^{m\times d_{v_{nh}}})\\
d_v = \sum_{i=0}^{nh}d_{v_i} \\
Q_i = x^{m\times h}\times W_Q^{h, d_{k_i}} \\
K_i = x^{m\times h}\times W_K^{h, d_{k_i}} \\
V_i = x^{m\times h}\times W_V^{h, d_{v_i}} \\
$$
即有$nh$个head。一般而言，为了保证参数量的大小不变，有 $d_{k_i}=d_{v_i} = \frac{h}{nh}$，所以多头head和单头head的权重大小一般是相同的，只是是否对其编码维度进行分割。

```python
class Attention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim) ## 这里多头没有体现在权重上，权重还是和单头一样
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def scaled_dot_product_attention(self, query, key, value, mask=None):

        """
        query: (bs, num_heads, seq_len, head_dim) 
        key: (bs, num_heads, seq_len, head_dim)
        value: (bs, num_heads, seq_len, head_dim)

        只在最后两个维度上做attention，即每个bs和head是独立的
        """

        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.nn.functional.softmax(scores, dim=-1)

        return torch.matmul(attention, value)
    

    def forward(self, x, mask=None):
        """
        x: (bs, seq_len, embed_dim)
        """

        bs, seq_len, embed_dim = x.shape

        query = self.q_proj(x) ## (bs, seq_len, embed_dim)，多头在这里没有体现
        key = self.k_proj(x)
        value = self.v_proj(x)


        ## 这里多头体现在了query, key, value的第二个维度上，即将最后一个维度分成了num_heads个head_dim，然后将每个head和bs放在一起
        query = query.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
        key = key.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention = self.scaled_dot_product_attention(query, key, value, mask)

        attention = attention.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        output = self.out_proj(attention)
        return output
    
```


## Flash Attention
用于优化Attention的实际实现，在原版的Attention计算中，分为以下几步：
$$
S = \frac{QK^T}{\sqrt{d_k}}\\
P = \text{softmax}(S)\\
O = PV
$$
上面一共需要三次的GPU Kerenel的启动，并产生了$S$和$P$两个中间结果的读写，造成了内存带宽的浪费。为了消除中间结果的产生，可以使用Kernel Fusion将上面三个操作融合在一起，一次性产生最后的结果。

在做矩阵乘法时，一般使用Tiling的操作，对矩阵进行分块计算。但是 $softmax()$ 需要对矩阵的一整行进行操作。**因此，FlashAttention的重点就是逐块进行 $softmax()$ 计算，并不断对前面的计算结果进行矫正，最后得到正确的结果。**

对于$X = \{x_1, x_2, \cdots, x_n\} = (X_1, X_2)$，进行 $softmax()$ 操作：
$$
m(X) = \max (X) = \max \{m(X_1), m(X_2)\}\\
f(X) = \{ e^{x_1 - m(X)}, e^{x_2 - m(X)}, \cdots, e^{x_n - m(X)} \} = \{  f(X_1)e^{m(X_1) - m(X)}, f(X_2)e^{m(X_2) - m(X)} \} \\
l(X) = \sum f(X) = e^{m(X_1) - m(X)}\sum f(X_1) +  e^{m(X_2) - m(X)}\sum f(X_2) = e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2) \\

\text{softmax}(X) = \frac{f(X)}{l(X)} = \frac{\{  f(X_1)e^{m(X_1) - m(X)}, f(X_2)e^{m(X_2) - m(X)} \}}{ e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2)} \\= \{\frac{  f(X_1)e^{m(X_1) - m(X)} }{ e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2)}, \frac{ f(X_2)e^{m(X_2) - m(X)} }{ e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2)}\} \\= \{\frac{  f(X_1) }{ l(X_1)} *( l(X_1) e^{m(X_1) - m(X)} ) , \frac{ f(X_2) }{l(X_2)} * ( l(X_2) e^{m(X_2) - m(X)} ) \} * (e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2))^{-1} \\ = \{\text{softmax} (X_1) *( l(X_1) e^{m(X_1) - m(X)} ) , \text{softmax} (X_2) * ( l(X_2) e^{m(X_2) - m(X)} ) \} * (e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2))^{-1} 
$$

通过上面的公式可以知道，在计算完 $\text{softmax} (X_1)$ 后，拿到$X_2$时，可以通过记录的 $l(X_1),m(X_1)$ 和最新的 $l(X_2),m(X_2)$ 对前面的计算结果进行矫正，得到正确的结果。

在实际计算Attention时，有
$$
O = \text{softmax}(S)V \\= \text{softmax}(S_1, S_2) * (V_1, V_2)^T \\= \{\text{softmax} (X_1) *( l(X_1) e^{m(X_1) - m(X)} ) , \text{softmax} (X_2) * ( l(X_2) e^{m(X_2) - m(X)} ) \} * (e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2))^{-1} * (V_1, V_2)^T \\= (\text{softmax} (X_1) *( l(X_1) e^{m(X_1) - m(X)} )*V_1^{T} + \text{softmax} (X_2) * ( l(X_2) e^{m(X_2) - m(X)} )*V_2^{T})*(e^{m(X_1) - m(X)}l(X_1) +  e^{m(X_2) - m(X)}l(X_2))^{-1}
$$
开始，只有 $S_1$ 和 $V_1$ 部分，得到
$$
O_1 = \text{softmax}(S_1)V_1\\
\text{record:} \quad  m(S_1), l(S_1)
$$
再次计算得到 $S_2$，并载入 $V_2$，有
$$
m(S) = \max \{m(S_1), m(S_2)\} \\
l(S) = e^{m(S_1) - m(S)}l(S_1) +  e^{m(S_2) - m(S)}l(S_2)
$$

就可以更新原来的$O_1$，即
$$
O_1 = [O_1*e^{m(S_1) - m(S)}*l(S_1) + \text{softmax}(S_2)V_2*e^{m(S_2) - m(S)}*l(S_2)] * l(S)^{-1}\\
=>O = O_1
$$
当然，实际上并不需要计算$\text{softmax}(S_2)$，因为已经有了正确的$m(S)$和$l(X)$，即可以直接计算正确的后半部分
$$
O = [O_1*e^{m(S_1) - m(S)}*l(S_1) + \hat{\text{exp}}(S_2)V_2] * l(S)^{-1}\\
\hat{\text{exp}}(S_2) = \{e^{s_1^2-m(S)}, e^{s_2^2-m(S)}, \cdots, e^{s_{|S_2|}^2-m(S)}\}
$$

因此，当有多个分块，即 $S = (S_1, S_2, \cdots , S_T ), V = (V_1, V_2, \cdots , V_T )$，初始化
$$
O=0\\
m=0\\
l=0
$$
每当有了 $(S_i, V_i)$，就可以更新上一次的计算结果 $O, l, m$，即
$$
m^{new} = \max \{m, m(S_i)\}\\
l^{new} = e^{m - m^{new}}l +  e^{m(S_i) - m^{new}}l(S_i) \\
O = [O*e^{m - m^{new}}*l + \hat{\text{exp}}(S_i)V_i] * (l^{new})^{-1}\\
\hat{\text{exp}}(S_i) = \{e^{s_1^i-m^{new}}, e^{s_2^i-m^{new}}, \cdots, e^{s_{|S_i|}^i-m^{new}}\}\\
l = l^{new}, m = m^{new}
$$

**FlashAttention2:** 在原本基础上，主要的优化点是调整了$Q,V,K$的循环顺序，减少了不必要的计算。具体而言，原本**FlashAttention**中$V,K$是外循环，$Q$是内循环；而**FlashAttention2**则将$Q$变成了外循环，$V,K$变成了内循环，主要是为了在seqlen维度也实现并行，而不是**FlashAttention**中的只有batch和head维度。然后去掉不必要的scale计算，即在计算过程中不将 $l$ 计算到结果中，只在最后一次更新，即
$$
m^{new} = \max \{m, m(S_i)\}\\
l^{new} = e^{m - m^{new}}l +  e^{m(S_i) - m^{new}}l(S_i) \\
O = O*e^{m - m^{new}} + \hat{\text{exp}}(S_i)V_i\\
\hat{\text{exp}}(S_i) = \{e^{s_1^i-m^{new}}, e^{s_2^i-m^{new}}, \cdots, e^{s_{|S_i|}^i-m^{new}}\}\\
l = l^{new}, m = m^{new}
$$ 
最后循环结束时，加入$l$，即
$$
O = O*l^{-1}
$$


**实现：**
```python
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
```

上面代码的重点就是其中的`for`循环，即每次加载一部分的`k,v`进行计算。
```python
for start_n in range(0, end_n, BLOCK_N):
```
经过和最原始attention的进行比较测试。可以看到，目前的实现需要`seqlen`比较长，才有效果。当然，目前的实现是没有任何调优的，但是当`seqlen`很大时，还是有明显的`3`倍加速。
```
bs: 1, (sq_q, sq_k): (1, 16), flash_time: 7.976293563842774e-05, basic_time: 5.3310394287109374e-05, speedup: 0.6683604842325511
bs: 1, (sq_q, sq_k): (1, 128), flash_time: 5.701780319213867e-05, basic_time: 5.52821159362793e-05, speedup: 0.9695588542755593
bs: 1, (sq_q, sq_k): (1, 256), flash_time: 4.194259643554687e-05, basic_time: 5.680084228515625e-05, speedup: 1.3542519326966804
bs: 1, (sq_q, sq_k): (1, 512), flash_time: 8.13126564025879e-05, basic_time: 6.02865219116211e-05, speedup: 0.7414162146312857
bs: 1, (sq_q, sq_k): (16, 16), flash_time: 5.601644515991211e-05, basic_time: 5.3637027740478515e-05, speedup: 0.9575228772079166
bs: 1, (sq_q, sq_k): (128, 128), flash_time: 5.697011947631836e-05, basic_time: 5.8808326721191405e-05, speedup: 1.0322661644695543
bs: 1, (sq_q, sq_k): (512, 512), flash_time: 8.419513702392578e-05, basic_time: 0.00017155885696411134, speedup: 2.0376337996262106
bs: 1, (sq_q, sq_k): (1024, 1024), flash_time: 0.00034065961837768553, basic_time: 0.001051325798034668, speedup: 3.086147407319275
bs: 4, (sq_q, sq_k): (1, 16), flash_time: 5.6281089782714844e-05, basic_time: 5.365133285522461e-05, speedup: 0.9532745912056256
bs: 4, (sq_q, sq_k): (1, 128), flash_time: 5.7156085968017576e-05, basic_time: 5.6073665618896484e-05, speedup: 0.9810620281149627
bs: 4, (sq_q, sq_k): (1, 256), flash_time: 4.2803287506103516e-05, basic_time: 6.133556365966797e-05, speedup: 1.4329638500529158
bs: 4, (sq_q, sq_k): (1, 512), flash_time: 8.398771286010742e-05, basic_time: 6.694316864013671e-05, speedup: 0.7970590740057342
bs: 4, (sq_q, sq_k): (16, 16), flash_time: 5.620718002319336e-05, basic_time: 5.433082580566406e-05, speedup: 0.9666171792152704
bs: 4, (sq_q, sq_k): (128, 128), flash_time: 5.9051513671875e-05, basic_time: 7.576465606689452e-05, speedup: 1.2830264857881135
bs: 4, (sq_q, sq_k): (512, 512), flash_time: 0.00033771514892578125, basic_time: 0.0010627198219299316, speedup: 3.1467934598441207
bs: 4, (sq_q, sq_k): (1024, 1024), flash_time: 0.0011913704872131348, basic_time: 0.0041971755027770995, speedup: 3.522980926441423
bs: 8, (sq_q, sq_k): (1, 16), flash_time: 4.187345504760742e-05, basic_time: 5.3539276123046876e-05, speedup: 1.278597050617776
bs: 8, (sq_q, sq_k): (1, 128), flash_time: 4.3766498565673826e-05, basic_time: 5.999088287353516e-05, speedup: 1.3707032739554394
bs: 8, (sq_q, sq_k): (1, 256), flash_time: 8.449316024780273e-05, basic_time: 6.577968597412109e-05, speedup: 0.7785208386241147
bs: 8, (sq_q, sq_k): (1, 512), flash_time: 0.00016553401947021483, basic_time: 7.947444915771484e-05, speedup: 0.4801094627682558
bs: 8, (sq_q, sq_k): (16, 16), flash_time: 4.197120666503906e-05, basic_time: 5.5022239685058594e-05, speedup: 1.3109520563508295
bs: 8, (sq_q, sq_k): (128, 128), flash_time: 7.083415985107422e-05, basic_time: 0.00011976957321166993, speedup: 1.690844833389431
bs: 8, (sq_q, sq_k): (512, 512), flash_time: 0.0006548142433166503, basic_time: 0.0022452139854431154, speedup: 3.428780006481
bs: 8, (sq_q, sq_k): (1024, 1024), flash_time: 0.002338559627532959, basic_time: 0.00832641839981079, speedup: 3.5604900990250425
bs: 16, (sq_q, sq_k): (1, 16), flash_time: 8.386850357055664e-05, basic_time: 5.439043045043945e-05, speedup: 0.648520339994883
bs: 16, (sq_q, sq_k): (1, 128), flash_time: 8.779764175415039e-05, basic_time: 6.557226181030273e-05, speedup: 0.7468567549219279
bs: 16, (sq_q, sq_k): (1, 256), flash_time: 0.00016885757446289063, basic_time: 7.869482040405273e-05, speedup: 0.4660425844346549
bs: 16, (sq_q, sq_k): (1, 512), flash_time: 0.00037823915481567384, basic_time: 0.00034990787506103513, speedup: 0.9250969144946263
bs: 16, (sq_q, sq_k): (16, 16), flash_time: 8.401393890380859e-05, basic_time: 5.720853805541992e-05, speedup: 0.6809410295703502
bs: 16, (sq_q, sq_k): (128, 128), flash_time: 0.00011700868606567382, basic_time: 0.00025438547134399415, speedup: 2.174073394869287
bs: 16, (sq_q, sq_k): (512, 512), flash_time: 0.0012639284133911133, basic_time: 0.004472081661224365, speedup: 3.5382396770603437
bs: 16, (sq_q, sq_k): (1024, 1024), flash_time: 0.004626307487487793, basic_time: 0.016603732109069826, speedup: 3.5889815266226694
bs: 32, (sq_q, sq_k): (1, 16), flash_time: 0.00016762256622314452, basic_time: 5.6202411651611326e-05, speedup: 0.33529144027536767
bs: 32, (sq_q, sq_k): (1, 128), flash_time: 0.00017493486404418946, basic_time: 7.828474044799805e-05, speedup: 0.4475079388876017
bs: 32, (sq_q, sq_k): (1, 256), flash_time: 0.00038260936737060546, basic_time: 0.0003447556495666504, speedup: 0.9010643203429753
bs: 32, (sq_q, sq_k): (1, 512), flash_time: 0.0006938076019287109, basic_time: 0.000688018798828125, speedup: 0.9916564720759853
bs: 32, (sq_q, sq_k): (16, 16), flash_time: 0.0001679682731628418, basic_time: 6.109952926635742e-05, speedup: 0.36375636967537717
bs: 32, (sq_q, sq_k): (128, 128), flash_time: 0.00034769535064697264, basic_time: 0.0006108474731445312, speedup: 1.7568468258430818
bs: 32, (sq_q, sq_k): (512, 512), flash_time: 0.0024597668647766114, basic_time: 0.008896472454071045, speedup: 3.616794982267149
bs: 32, (sq_q, sq_k): (1024, 1024), flash_time: 0.009153897762298585, basic_time: 0.033102684020996094, speedup: 3.616239210943936
```





## Ring Attention



## 参考
https://transformers.run/c1/attention/          
https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html     

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
https://triton-lang.org/main/python-api/generated/triton.language.trans.html#triton.language.trans
