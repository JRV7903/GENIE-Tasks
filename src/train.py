import torch
import logging

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
            
    return running_loss / len(dataloader.dataset)
