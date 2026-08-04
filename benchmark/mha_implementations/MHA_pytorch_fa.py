import torch
from torch import nn

class MHA_pytorch_fa(nn.Module):
    def __init__(self, dims: int, heads: int, p: float = 0.0, if_bias: bool = False, need_weights=False) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dims,
            num_heads=heads,
            dropout=p,
            bias=if_bias,
            add_bias_kv=if_bias,
            batch_first=True,
        )

        self.need_weights = need_weights  # 是否启用SPDA
        self.proj = nn.Linear(dims, dims, bias=if_bias)

    def forward(self, x: torch.Tensor):
        # x shape (B, L, d)
        B, L, d= x.shape

        # attention
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=self.need_weights)

        # proj
        out = self.proj(out)
        return out

if __name__ == "__main__":
    batch_size = 8
    seq_len = 1024
    embed_dim = 768

    models = MHA_pytorch_fa(
        dims=embed_dim,
        heads=12,
        p=0.0,
        if_bias=False
    )

    embeddings = torch.rand((batch_size, seq_len, embed_dim))
    out = models(embeddings)
    print(out.shape)


