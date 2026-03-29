import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_loss_curve(train_losses, val_losses=None, save_path=None):
    """Monitors learning stability across training."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_reconstruction(original, reconstructed, save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    channels = ['ECAL', 'HCAL', 'Tracks']
    
    orig = original.cpu().numpy()
    recon = reconstructed.detach().cpu().numpy()
    
    for i in range(3):
        # Original
        ax_orig = axes[i, 0]
        im_orig = ax_orig.imshow(orig[i], cmap='viridis')
        ax_orig.set_title(f'Original {channels[i]}')
        ax_orig.axis('off')
        
        # Recon
        ax_recon = axes[i, 1]
        im_recon = ax_recon.imshow(recon[i], cmap='viridis')
        ax_recon.set_title(f'Reconstructed {channels[i]}')
        ax_recon.axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_comparison(baseline_results, nonlocal_results, save_path=None):
    plt.figure(figsize=(8, 6))
    
    plt.plot(baseline_results['fpr'], baseline_results['tpr'], 
             label=f"Baseline GNN (AUC={baseline_results['auc']:.4f})")
    
    plt.plot(nonlocal_results['fpr'], nonlocal_results['tpr'], 
             label=f"Non-Local GNN (AUC={nonlocal_results['auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('GNN Performance Comparison')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()
