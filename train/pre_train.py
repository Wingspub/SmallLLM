from dataset.dataset import PretrainDataset
from torch.utils.data import DataLoader
from torch import optim, nn
from model.model import SLMConfig, SLMforCasualLM
import torch

# Argparse
lr = 1e-4
train_iter_num = 10000
batch_size = 12
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dataset
train_dataset = PretrainDataset()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model
config = SLMConfig()
model = SLMforCasualLM(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

# Train
step = 0
for data in train_dataloader:
    y_pred = model(data)
    loss = loss_func(y_pred, data)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

