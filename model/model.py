from typing import Any

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin

class SmallLLMConfig(PretrainedConfig):
    def __init__(self, ) -> None:
        super().__init__()


class Attention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class SmallTransformer(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)


class SmallLMforCasualLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)



