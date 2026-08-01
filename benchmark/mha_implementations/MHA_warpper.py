import torch
from torch import nn
from typing import cast


class CasualAttention(nn.Module):
    """单头注意力"""
    def __init__(self, in_dims: int, out_dims: int, p: float, if_bias: bool) -> None:
        super().__init__()
        self.W_Q = nn.Linear(in_dims, out_dims, bias=if_bias)
        self.W_K = nn.Linear(in_dims, out_dims, bias=if_bias)
        self.W_V = nn.Linear(in_dims, out_dims, bias=if_bias)
        self.dropout = nn.Dropout(p)


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, L, d)
        queries = cast(torch.Tensor, self.W_Q(x))
        keys = cast(torch.Tensor, self.W_K(x))
        values = cast(torch.Tensor, self.W_V(x))

        # Attention
        score = queries @ keys.transpose(1, 2) * (queries.shape[-1]) ** (-0.5)
        score = torch.softmax(score + mask, dim=-1)
        out = score @ values

        return out


class MHAwithWarpper(nn.Module):
    """组装多个单头注意力为多头注意力"""
    def __init__(self, dims: int, heads: int, p: float = 0.0, if_bias: bool = False) -> None:
        super().__init__()
        heads_num = dims // heads
        assert dims  == heads_num * heads

        self.attentions = nn.ModuleList([
            CasualAttention(
                in_dims=dims,
                out_dims=heads_num,
                p=p,
                if_bias=if_bias
            )
            for _ in range(heads)
        ])

        self.proj = nn.Linear(dims, dims, bias=if_bias)

    def forward(self, x: torch.Tensor):
        # x: (B, L, d)
        length = x.shape[1]
        mask = -torch.log(torch.triu(torch.ones(length, length, device=x.device), diagonal=1))
        out = torch.cat([attention(x, mask) for attention in self.attentions], dim=-1)
        return self.proj(out)


if __name__ == "__main__":
    batch_size = 8
    seq_len = 1024
    embed_dim = 768

    models = MHAwithWarpper(
        dims=embed_dim,
        heads=12,
        p=0.0,
        if_bias=False
    )

    embeddings = torch.rand((batch_size, seq_len, embed_dim))
    out = models(embeddings)
    print(out.shape)
