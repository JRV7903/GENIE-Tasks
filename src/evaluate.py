import torch

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            running_loss += loss.item() * data.size(0)
            
    return running_loss / len(dataloader.dataset)
