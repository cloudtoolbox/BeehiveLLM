import json

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Build input text
        # Use tokenizer to convert text to token ids
        # Force padding to max_length, for example: input_ids = [50256, 100, 200, ..., 0, 0, 0]
        # return_tensors='pt' → returns a PyTorch tensor
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Build loss mask:
        # pad token → mask = False (0)
        # non-pad   → mask = True (1)
        # Used to ignore padding when computing loss.
        input_ids = encoding.input_ids.squeeze().to(torch.long)
        loss_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)

        X = input_ids[:-1].clone()
        Y = input_ids[1:].clone()
        loss_mask = loss_mask[1:].clone()

        return X, Y, loss_mask
