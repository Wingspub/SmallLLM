from typing import Tuple, Sequence
from torch.utils.data import Dataset
from transformers import TokenizersBackend
from datasets import load_dataset
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_files: Sequence[str], tokenizer: TokenizersBackend, max_length: int) -> None:
        super().__init__()
        dataset = load_dataset("json", data_files=data_files, split="train")
        text_list = [text for text in dataset['text']]
        encodings = tokenizer(text=text_list, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.encodings = encodings


    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encodings["input_ids"][index], self.encodings["attention_mask"][index]


class SFTDataset(Dataset):
    ...

