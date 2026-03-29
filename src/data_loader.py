import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class JetDataset(Dataset):
    def __init__(self, h5_filepath, normalize=True, max_samples=-1):
        self.normalize = normalize
        print(f"Loading {max_samples} images...")
        with h5py.File(h5_filepath, 'r') as f:
            total = len(f['X_jets'])
            num = min(max_samples, total) if max_samples > 0 else total
            self.images = f['X_jets'][:num]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # (125, 125, 3) -> (3, 125, 125)
        img = self.images[idx].transpose(2, 0, 1)
        tensor = torch.tensor(img, dtype=torch.float32)

        if self.normalize:
            max_val = tensor.max()
            if max_val > 0:
                tensor = tensor / max_val

        return tensor

def get_dataloaders(filepath, batch_size, num_workers=0, normalize=True, train_ratio=0.8, max_samples=-1):
    dataset = JetDataset(filepath, normalize=normalize, max_samples=max_samples)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
