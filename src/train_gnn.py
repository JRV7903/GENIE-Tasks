import torch
import torch.nn as nn
from tqdm import tqdm

def train_gnn_epoch(model, dataloader, optimizer, device, criterion=None):
    model.train()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_samples = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch).squeeze(-1)
        y = batch.y.to(torch.float32)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_samples
