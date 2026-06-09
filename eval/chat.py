# from transformers import AutoModel
from model.model_SLM import SLMforCasualLM

model = SLMforCasualLM.from_pretrained("./checkpoint")
print(model)

