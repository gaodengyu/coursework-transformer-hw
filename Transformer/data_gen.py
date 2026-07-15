#Data generation
import random
import torch
from torch.utils.data import Dataset, DataLoader

class AdditionDataset(Dataset):
    def __init__(self, num_samples, max_len=16):
        self.num_samples = num_samples
        self.max_len = max_len
        
        self.chars = " 0123456789+=" 
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)} 
        self.itos = {i: ch for i, ch in enumerate(self.chars)} 
        
        self.data = self._generate_data()

    def _generate_data(self):
        samples = []
        for _ in range(self.num_samples):
            a = random.randint(0, 999)
            b = random.randint(0, 999)
            res = a + b
            s = f"{a}+{b}={res}"
            if len(s) < self.max_len:
                s = s + " " * (self.max_len - len(s))
            else:
                s = s[:self.max_len]
            indices = [self.stoi[c] for c in s]
            samples.append(torch.tensor(indices, dtype=torch.long))
            
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = self.data[idx]
        x = data[:-1]
        y = data[1:]
        return x, y

train_dataset = AdditionDataset(num_samples=20000) # 训练集 20,000 条
val_dataset = AdditionDataset(num_samples=2000)   # 验证集 2,000 条

sample_x, sample_y = train_dataset[0]
print(f"字符表大小: {train_dataset.vocab_size}")
print(f"输入序列 (indices): {sample_x}")
print(f"输入序列 (解码): {''.join([train_dataset.itos[i.item()] for i in sample_x])}")