from typing import cast
from dataset.dataset import PretrainDataset
from torch.utils.data import DataLoader
from torch import optim, nn
from transformers import TokenizersBackend
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.model_SLM import SLMConfig, SLMforCasualLM
import torch

# Argparse
lr = 1e-4
train_iter_num = 10000
batch_size = 8
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

# Dataset
data_files = ["./data/pretrain_data.jsonl"]
tokenizer:TokenizersBackend = TokenizersBackend.from_pretrained("Qwen/Qwen2.5-7B")
train_dataset = PretrainDataset(data_files, tokenizer, 128)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model
vocab_size = tokenizer.vocab_size + 22
config = SLMConfig(vocab_size=vocab_size, hidden_size=512, num_hidden_layers=8, num_head=4)
model = SLMforCasualLM(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train
step = 0
for _ in range(train_iter_num):
    for data, attention_mask in train_dataloader:

        data = data.to(device)
        output: CausalLMOutputWithPast = model(data, attention_mask)
        loss = cast(torch.FloatTensor, output.loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"{step} loss:{loss.cpu().item():.6f}")

        step += 1
        if step % 1000 == 0:
            pretext = tokenizer("随着", return_tensors="pt")["input_ids"].to(device)
            output_ids = model.generate(pretext, max_length=128).cpu()
            print(tokenizer.decode(output_ids))
