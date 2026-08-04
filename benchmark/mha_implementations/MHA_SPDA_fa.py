import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from typing import cast

class MHA_SPDA_fa(nn.Module):
    def __init__(self, dims: int, heads: int, p: float = 0.0, if_bias: bool = False) -> None:
        super().__init__()

        head_dim = dims // heads
        assert dims == heads * head_dim
        self.heads = heads
        self.p = p

        self.W_Q = nn.Linear(dims, dims, bias=if_bias)
        self.W_K = nn.Linear(dims, dims, bias=if_bias)
        self.W_V = nn.Linear(dims, dims, bias=if_bias)

        # self.dropout = nn.Dropout(p=p)
        self.proj = nn.Linear(dims, dims, bias=if_bias)

    def forward(self, x: torch.Tensor):
        # x (B, L, d)
        B, L, d = x.shape

        queris = cast(torch.Tensor, self.W_Q(x)).view(B, L, self.heads, -1).transpose(1, 2)
        keys = cast(torch.Tensor, self.W_K(x)).view(B, L, self.heads, -1).transpose(1, 2)
        values = cast(torch.Tensor, self.W_V(x)).view(B, L, self.heads, -1).transpose(1, 2)

        # attention
        # mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        use_dropout = 0. if not self.training else self.p
        out = scaled_dot_product_attention(
            query=queris,
            key=keys,
            value=values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, L, d)

        out = self.proj(out)
        return out

if __name__ == "__main__":
    batch_size = 8
    seq_len = 1024
    embed_dim = 768

    models = MHA_SPDA_fa(
        dims=embed_dim,
        heads=12,
        p=0.0,
        if_bias=False
    )

    embeddings = torch.rand((batch_size, seq_len, embed_dim))
    out = models(embeddings)
    print(out.shape)