from typing import cast
from .utils import model_init
from dataset.dataset import PretrainDataset
from torch.utils.data import DataLoader
from torch import optim
from transformers import TokenizersBackend
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import time
import os

# Argparse
lr = 1e-3
train_iter_num = 10000
save_iter_num = 10000
save_dir = "./checkpoint"
batch_size = 16
seq_len = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dataset
data_files = ["./data/pretrain_data.jsonl"]
tokenizer:TokenizersBackend = TokenizersBackend.from_pretrained("Qwen/Qwen2.5-7B")
train_dataset = PretrainDataset(data_files, tokenizer, seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model
model = model_init()
model = model.to(device).to(torch.bfloat16)
model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train
model.train()
step = 0
s1 = time.perf_counter()
for _ in range(train_iter_num):
    for input_ids, labels in train_dataloader:
        step += 1
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        output: CausalLMOutputWithPast = model(input_ids=input_ids, labels=labels)
        loss = cast(torch.FloatTensor, output.loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print the loss
        if step % 50 == 0:
            s2 = time.perf_counter()
            print(f"{step} loss:{loss.cpu().item():.6f}, cost time:{s2-s1:.2f}s")
            s1 = time.perf_counter()

        # print the generate
        if step % 1000 == 0:
            model.eval()
            pretext = tokenizer("随着", return_tensors="pt")["input_ids"].to(device)
            prefill_len = len(pretext)
            gs1 = time.perf_counter()
            output_ids = model.generate(pretext, max_length=seq_len, do_sample=True, num_return_sequences=4, temperature=0.8, top_p=0.9).cpu()
            gs2 = time.perf_counter()
            print(f"decode time: {(seq_len-prefill_len)/(gs2-gs1):.2f} tokens/s")
            for i, ids in enumerate(output_ids):
                print(i, tokenizer.decode(ids))
            # greedy sample
            output_cache_ids = model.generate(pretext, max_length=seq_len).cpu()
            print("greedy sample:", tokenizer.decode(output_cache_ids)[0])
            s1 = time.perf_counter()
        

        # save the model
        if step % save_iter_num == 0:
            torch.save(model, save_dir+"/pretrain.pt")
