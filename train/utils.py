from model.config_SLM import CONFIG
from model.model_SLM import SLMforCasualLM


def model_init(model_type: str="test") -> SLMforCasualLM:
    '''Model Load'''
    config = CONFIG[model_type]
    model = SLMforCasualLM(config)
    return model


