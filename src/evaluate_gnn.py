import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def evaluate_gnn_model(model, dataloader, device, return_predictions=False):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    num_samples = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            y = batch.y.to(torch.float32)

            loss = criterion(out, y)
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

            all_preds.extend(torch.sigmoid(out).cpu().numpy())
            all_true.extend(y.cpu().numpy())

    auc = roc_auc_score(all_true, all_preds) if len(set(all_true)) > 1 else 0.5

    if return_predictions:
        return total_loss / num_samples, auc, all_true, all_preds

    return total_loss / num_samples, auc
