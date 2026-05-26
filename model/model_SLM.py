from typing import Any, cast

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
    is_casual: bool

    def __init__(
        self,
        vocab_size: int = 3600,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_head: int = 4,
        is_casual: bool = True
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_head = num_head

        head_hidden_size = hidden_size // num_head
        assert head_hidden_size * num_head == hidden_size

        self.is_casual = is_casual


class FFN(nn.Module):
    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.up_proj = nn.Linear(hidden_size, 2*hidden_size)
        self.gate = nn.Linear(hidden_size, 2*hidden_size)
        self.down_proj = nn.Linear(2*hidden_size, hidden_size)


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

        self.WQ = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.WV = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.WK = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, input_embs: torch.Tensor) -> torch.Tensor:
        B, L, d = input_embs.shape
        Q = cast(torch.Tensor, self.WQ(input_embs)).reshape(B, L, self.num_head, self.head_hidden_size).transpose(1, 2)
        K = cast(torch.Tensor, self.WK(input_embs)).reshape(B, L, self.num_head, self.head_hidden_size).transpose(1, 2)
        V = cast(torch.Tensor, self.WV(input_embs)).reshape(B, L, self.num_head, self.head_hidden_size).transpose(1, 2)

        # A = QK^T
        A = torch.matmul(Q, K.transpose(-1, 2)) / (self.head_hidden_size) ** 0.5
        if self.is_casual: A += torch.log(torch.tril(torch.ones_like(A)))
        output = torch.matmul(F.softmax(A, dim=-1), V)
        output = output.transpose(1, 2).reshape(B, L, d)

        return output


class AttentionBlock(nn.Module):
    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.FFN = FFN(config)
        self.attention_dropout = nn.Dropout()
        self.residual_dropout = nn.Dropout()


    def forward(self, input_ids_embs: torch.Tensor) -> torch.Tensor:
        # Attention
        embs = self.attention(input_ids_embs)
        input_ids_embs += embs

        # FFN
        embs = self.FFN(input_ids_embs)
        input_ids_embs += embs

        return input_ids_embs


class SLMModel(PreTrainedModel):
    config_class = SLMConfig
    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            layer = AttentionBlock(config=config)
            self.layers.append(layer)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embs = self.embed_tokens(input_ids)
        for layer in self.layers:
            embs = layer(embs)

        return embs


class SLMforCasualLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(self, input_ids: torch.Tensor, past_key_values: Cache | None = None, use_cache: bool = False, return_dict: bool = False) -> CausalLMOutputWithPast:
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
    output = model(input_ids)
    print(output)

    gen_res = model.generate(input_ids, max_length=256)
    print(gen_res)
