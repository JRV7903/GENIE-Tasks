import torch
import numpy as np
import h5py
from torch_geometric.data import Data, Dataset as PyGDataset
from tqdm import tqdm

def get_edge_data(x, k=12):
    if x.size(0) <= 1: 
        return torch.empty((2, 0), dtype=torch.long, device=x.device), torch.empty(0, device=x.device)
    
    k = min(k, x.size(0) - 1)
    dist = torch.cdist(x, x)
    val, indices = dist.topk(k + 1, dim=1, largest=False)
    
    # Distance-based weights
    weights = 1.0 / (1.0 + val[:, 1:].reshape(-1))
    
    indices = indices[:, 1:]
    row = torch.arange(x.size(0), device=x.device).view(-1, 1).repeat(1, k).view(-1)
    edge_index = torch.stack([row, indices.reshape(-1)], dim=0)
    
    return edge_index, weights

class AdvancedJetGraphDataset(PyGDataset):
    def __init__(self, h5_filepath, k_neighbors=12, max_samples=-1):
        self.graphs = []
        print(f"Loading {max_samples} graphs...")
        
        with h5py.File(h5_filepath, 'r') as f:
            total = len(f['X_jets'])
            num = min(max_samples, total) if max_samples > 0 else total
            
            # (N, 125, 125, 3) -> (N, 3, 125, 125)
            images = f['X_jets'][:num].transpose(0, 3, 1, 2)
            labels = f['y'][:num].astype(np.int64)
            
            for i in tqdm(range(0, num, 5000), desc="Parsing Dataset"):
                batch_images = images[i:min(i+5000, num)]
                batch_labels = labels[i:min(i+5000, num)]
                
                b, c, y, x = np.nonzero(batch_images)
                vals = batch_images[b, c, y, x]
                pcs_flat = np.stack((x, y, vals, c), axis=1)
                
                counts = np.bincount(b, minlength=batch_images.shape[0])
                splits = np.cumsum(counts)[:-1]
                pcs_split = np.split(pcs_flat, splits)
                
                for idx, pc in enumerate(pcs_split):
                    if len(pc) > 400:
                        sort_idx = np.argsort(pc[:, 2])[-400:]
                        pc = pc[sort_idx]
                    if len(pc) > 0:
                        feat = torch.tensor(pc, dtype=torch.float32)
                        feat[:, :2] = (feat[:, :2] - 62.5) / 125.0
                        ei, ew = get_edge_data(feat[:, :2], k=k_neighbors)
                    else:
                        feat = torch.zeros((1, 4))
                        ei = torch.empty((2, 0), dtype=torch.long)
                        ew = torch.empty(0, dtype=torch.float32)
                    
                    self.graphs.append(
                        Data(
                            x=feat,
                            edge_index=ei,
                            edge_attr=ew,
                            y=torch.tensor(batch_labels[idx], dtype=torch.long)
                        )
                    )
        super().__init__(".")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
