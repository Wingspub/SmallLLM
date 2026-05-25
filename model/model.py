from typing import Any

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch


class SLMConfig(PretrainedConfig):
    model_type = "SmallLLM"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int

    def __init__(
        self,
        vocab_size: int=3600,
        hidden_size: int=1024,
        num_hidden_layers: int=6
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class Attention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class AttentionBlock(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class SLMModel(PreTrainedModel):
    config_class = SLMConfig
    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


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
        loss = nn.functional.cross_entropy(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1))

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
