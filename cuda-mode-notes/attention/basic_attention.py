import torch


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
    


if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    seq_len = 10
    bs = 32

    x = torch.randn(bs, seq_len, embed_dim)
    mask = torch.zeros(bs, 1, seq_len, seq_len)
    mask[:, :, 5:] = 1

    attention = Attention(embed_dim, num_heads)
    output = attention(x, mask)
    print(output.shape) ## torch.Size([32, 10, 512])