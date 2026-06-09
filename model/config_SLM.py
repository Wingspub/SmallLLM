from transformers import PretrainedConfig


class SLMConfig(PretrainedConfig):
    model_type = "SLM"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None
    p: float
    is_casual: bool

    def __init__(
        self,
        vocab_size: int = 3600,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
        num_key_value_heads: int | None = None,
        p: float = 0.0,
        is_casual: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.p = p

        head_hidden_size = hidden_size // num_attention_heads
        assert head_hidden_size * num_attention_heads == hidden_size

        self.is_casual = is_casual

tokenizer_vocab_size = 151665

# Model Recommendation Config
CONFIG = {
    "test": SLMConfig(
        vocab_size=tokenizer_vocab_size+22,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1
    ),
    "samll": SLMConfig(),
    "large": SLMConfig(),
    "custom": SLMConfig(), # you can define the model size you want
}

