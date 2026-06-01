from typing import Tuple, Sequence
from torch.utils.data import Dataset
from transformers import TokenizersBackend
from datasets import load_dataset
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_files: Sequence[str], tokenizer: TokenizersBackend, max_length: int) -> None:
        super().__init__()
        dataset = load_dataset("json", data_files=data_files, split="train")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self) -> int:
        return len(self.dataset['text'])


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        encodings = self.tokenizer(text=self.dataset['text'][index], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids: torch.Tensor = encodings["input_ids"][0]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    ...

