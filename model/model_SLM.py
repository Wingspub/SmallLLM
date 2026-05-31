from typing import Any, cast, Tuple

from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch


class SLMConfig(PretrainedConfig):
    model_type = "SLM"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_head: int
    p: float
    is_casual: bool

    def __init__(
        self,
        vocab_size: int = 3600,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_head: int = 4,
        p: float = 0.0,
        is_casual: bool = True
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_head = num_head
        self.p = p

        head_hidden_size = hidden_size // num_head
        assert head_hidden_size * num_head == hidden_size

        self.is_casual = is_casual


def precompute_rope_freqs(dim: int, max_seq_length: int = 32*1024, rope_base: float = 1e6) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs, atten_factor = 1.0/ (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)), 1.0
    t = torch.arange(max_seq_length, device=freqs.device)

    # Length * dim
    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cos(freqs.repeat(1, 2)) * atten_factor
    freqs_sin = torch.sin(freqs.repeat(1, 2)) * atten_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_weight: torch.Tensor, sin_weight: torch.Tensor, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin_weight.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin_weight.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))


    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


    def forward(self, input_seq_emb: torch.Tensor) -> torch.Tensor:
        return (self.weight * self.norm(input_seq_emb.float())).type_as(input_seq_emb)


class FFN(nn.Module):
    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = int(8*hidden_size/3)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


    def forward(self, input_embs: torch.Tensor) -> torch.Tensor:
        output = F.silu(self.gate(input_embs) * self.up_proj(input_embs))
        output = self.down_proj(output)

        return output


class Attention(nn.Module):
    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_head = config.num_head
        self.head_hidden_size = self.hidden_size // self.num_head
        self.is_casual = config.is_casual
        self.p = config.p


        self.WQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.WV = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.WK = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_hidden_size)
        self.k_norm = RMSNorm(self.head_hidden_size)

        self.score_dropout = nn.Dropout(p=self.p)
        self.residual_dropout = nn.Dropout(p=self.p)

        self.output_project = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


    def forward(self, input_embs: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, L, d = input_embs.shape
        Q = cast(torch.Tensor, self.WQ(input_embs)).reshape(B, L, self.num_head, self.head_hidden_size).transpose(1, 2)
        K = cast(torch.Tensor, self.WK(input_embs)).reshape(B, L, self.num_head, self.head_hidden_size).transpose(1, 2)
        V = cast(torch.Tensor, self.WV(input_embs)).reshape(B, L, self.num_head, self.head_hidden_size).transpose(1, 2)
        Q, K = cast(torch.Tensor, self.q_norm(Q)), cast(torch.Tensor, self.k_norm(K))

        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, unsqueeze_dim=0)

        # A = QK^T
        A = torch.matmul(Q, K.transpose(-1, 2)) / (self.head_hidden_size) ** 0.5
        if self.is_casual: A += torch.log(torch.tril(torch.ones_like(A)))

        output = torch.matmul(self.score_dropout(F.softmax(A, dim=-1)), V)
        output = output.transpose(1, 2).reshape(B, L, d)
        output = self.residual_dropout(self.output_project(output))
        return output


class AttentionBlock(nn.Module):
    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.LN1 = RMSNorm(config.hidden_size)
        self.FFN = FFN(config)
        self.LN2 = RMSNorm(config.hidden_size)


    def forward(self, input_ids_embs: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Attention
        normed = self.LN1(input_ids_embs)
        embs = self.attention(normed, position_embeddings=position_embeddings)
        x = input_ids_embs + embs

        # FFN
        normed = self.LN2(x)
        embs = self.FFN(normed)
        output_embs = x + embs

        return output_embs


class SLMModel(PreTrainedModel):
    config_class = SLMConfig
    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        head_dims = config.hidden_size // config.num_head
        freqs_cos, freqs_sin = precompute_rope_freqs(dim=head_dims)
        self.freqs_cos = nn.Buffer(freqs_cos, persistent=False)
        self.freqs_sin = nn.Buffer(freqs_sin, persistent=False)

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = AttentionBlock(config=config)
            self.layers.append(layer)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        L = input_ids.shape[1]
        embs = self.embed_tokens(input_ids)
        position_embeddings = (self.freqs_cos[:L], self.freqs_sin[:L])

        for layer in self.layers:
            embs = layer(embs, position_embeddings)

        return embs


class SLMforCasualLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        return_dict: bool = False
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        # CrossEntropy
        y_pred, y = logits[:, :-1], input_ids[:, 1:]
        loss = F.cross_entropy(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)


if __name__ == "__main__":
    '''Module Test'''
    vocab_size = 256
    hidden_size = 64
    num_hidden_layers = 6
    config = SLMConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    model = SLMforCasualLM(config=config)

    # input
    input_ids = torch.randint(0, vocab_size, (1, 128))
    attention_mask = torch.randint(0, vocab_size, (1, 128))
    output = model(input_ids, attention_mask)
    print(output)

    gen_res = model.generate(input_ids, max_length=256)
    print(gen_res)
