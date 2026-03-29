import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GNNDataLoader
from sklearn.metrics import roc_curve, roc_auc_score

from src.utils import setup_logging, load_config, create_directories
from src.data_loader import get_dataloaders
from src.graph_utils import AdvancedJetGraphDataset
from src.model import ConvAutoencoder, JetGraphClassifier
from src.train import train_epoch
from src.evaluate import evaluate_model
from src.train_gnn import train_gnn_epoch
from src.evaluate_gnn import evaluate_gnn_model
from src.visualization import plot_reconstruction, plot_roc_comparison

def run_task1_ae(config, device):
    print("\n--- Task 1: Auto-encoder Training ---")
    
    max_samples = config["data"].get("max_samples")
    if max_samples is None:
        max_samples = -1
        
    train_loader, val_loader = get_dataloaders(
        filepath=config["data"]["filepath"],
        batch_size=config["data"]["batch_size"],
        max_samples=max_samples
    )
    
    model = ConvAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()
    
    epochs = config["training"]["ae_epochs"]
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        train_losses.append(loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} | Train MSE: {loss:.6f} | Val MSE: {val_loss:.6f}")
        
    from src.visualization import plot_loss_curve
    plot_loss_curve(train_losses, val_losses, save_path="outputs/autoencoder_loss_curve.png")
    
    # Visualization
    sample_batch = next(iter(val_loader)).to(device)
    recon = model(sample_batch)
    plot_reconstruction(sample_batch[0], recon[0], save_path="outputs/task1_ae_recon.png")

def run_gnn_classification(config, device, use_non_local=False):
    desc = "Non-Local" if use_non_local else "Baseline"
    print(f"\n--- Task 2/4: {desc} GNN Training ---")
    
    max_samples = config["data"].get("max_samples")
    if max_samples is None:
        max_samples = -1
        
    dataset = AdvancedJetGraphDataset(config["data"]["filepath"], max_samples=max_samples)
    train_size = int(config["data"]["split_ratio"] * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = GNNDataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = GNNDataLoader(val_ds, batch_size=config["data"]["batch_size"], num_workers=4, pin_memory=True)
    
    model = JetGraphClassifier(use_non_local=use_non_local).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    epochs = config["training"]["gnn_epochs"]
    for epoch in range(1, epochs + 1):
        train_gnn_epoch(model, train_loader, optimizer, device, criterion)
        _, auc = evaluate_gnn_model(model, val_loader, device)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"outputs/{desc.lower()}_best.pth")
        print(f"Epoch {epoch} | Val AUC: {auc:.4f}")
        
    # Get ROC data for final comparison
    model.load_state_dict(torch.load(f"outputs/{desc.lower()}_best.pth"))
    _, auc, y_true, y_pred = evaluate_gnn_model(model, val_loader, device, return_predictions=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc}

def main():
    create_directories()
    setup_logging()
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Common Task 1
    run_task1_ae(config, device)
    
    # Common Task 2: Baseline
    baseline_res = run_gnn_classification(config, device, use_non_local=False)
    
    # Specific Task 4: Non-Local
    nonlocal_res = run_gnn_classification(config, device, use_non_local=True)
    
    # Performance Comparison
    plot_roc_comparison(baseline_res, nonlocal_res, save_path="outputs/gnn_comparison.png")

if __name__ == "__main__":
    main()
