import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder_fc = nn.Linear(64 * 16 * 16, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        latent = self.encoder_fc(x.reshape(x.size(0), -1))
        x = self.decoder_fc(latent).view(-1, 64, 16, 16)
        return self.decoder(x)

class ResidualGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, edge_weight):
        out = F.relu(self.bn(self.conv(x, edge_index, edge_weight)))
        return out + self.shortcut(x)

class JetGraphClassifier(nn.Module):
    def __init__(self, node_features=4, hidden_dim=64, use_non_local=False):
        super().__init__()
        self.use_non_local = use_non_local
        self.block1 = ResidualGCNBlock(node_features, hidden_dim)
        self.block2 = ResidualGCNBlock(hidden_dim, hidden_dim)
        self.block3 = ResidualGCNBlock(hidden_dim, hidden_dim * 2)
        
        # Self-attention based non-local layer
        if use_non_local:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.block1(x, edge_index, edge_attr)
        x = self.block2(x, edge_index, edge_attr)
        x = self.block3(x, edge_index, edge_attr)
        
        if self.use_non_local:
            from torch_geometric.utils import to_dense_batch
            dense_x, mask = to_dense_batch(x, batch)
            attn_out, _ = self.attention(dense_x, dense_x, dense_x, key_padding_mask=~mask)
            x = attn_out[mask]
        
        p1 = global_mean_pool(x, batch)
        p2 = global_max_pool(x, batch)
        combined = torch.cat([p1, p2], dim=1)
            
        return self.fc(combined)
