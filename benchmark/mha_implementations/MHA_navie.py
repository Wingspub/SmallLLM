from typing import cast

import torch
from torch import nn

class navieMHA(nn.Module):
    def __init__(self, dims: int, heads: int, p: float = 0.0, if_bias: bool = False) -> None:
        super().__init__()
        head_num = dims // heads
        self.heads = heads
        assert dims == head_num * heads

        self.W_Q = nn.Linear(dims, dims, bias=if_bias)
        self.W_K = nn.Linear(dims, dims, bias=if_bias)
        self.W_V = nn.Linear(dims, dims, bias=if_bias)
        self.dropout = nn.Dropout(p)

        self.proj = nn.Linear(dims, dims, bias=if_bias)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        B, L, d = x.shape
        queirs = cast(torch.Tensor, self.W_Q(x))
        keys = cast(torch.Tensor, self.W_K(x))
        values = cast(torch.Tensor, self.W_V(x))

        queirs = queirs.view(B, L, self.heads, -1).transpose(1, 2)
        keys = keys.view(B, L, self.heads, -1).transpose(1, 2)
        values = values.view(B, L ,self.heads, -1).transpose(1, 2)

        # Attention
        mask = -torch.log(torch.triu(torch.ones(L, L, device=x.device), diagonal=1))
        score = (queirs @ keys.transpose(2, 3)) * (queirs.shape[-1]) ** (-0.5)
        score = torch.softmax(score + mask, dim=-1)

        # proj
        out = (score @ values).transpose(1, 2).contiguous().view(B, L, d)
        out = self.proj(out)
        return out

if __name__ == "__main__":
    batch_size = 8
    seq_len = 1024
    embed_dim = 768

    models = navieMHA(
        dims=embed_dim,
        heads=12,
        p=0.0,
        if_bias=False
    )

    embeddings = torch.rand((batch_size, seq_len, embed_dim))
    out = models(embeddings)
    print(out.shape)
