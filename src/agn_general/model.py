"""
Generalized AGN Model Architecture
Graph Variational Autoencoder for generating new nodes and edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    """
    Graph encoder that learns node representations
    """
    def __init__(self, in_channels, hidden_channels, latent_dim, num_layers=2):
        super(GraphEncoder, self).__init__()
        self.num_layers = num_layers
        
        # First layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Hidden layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layers for mean and log variance
        self.mu_layer = GCNConv(hidden_channels, latent_dim)
        self.logvar_layer = GCNConv(hidden_channels, latent_dim)
        
    def forward(self, x, edge_index):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
        
        # Compute mean and log variance
        mu = self.mu_layer(x, edge_index)
        logvar = self.logvar_layer(x, edge_index)
        
        return mu, logvar

class NodeDecoder(nn.Module):
    """
    Decoder that generates node features from latent representation
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(NodeDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1] for normalized features
        )
    
    def forward(self, z):
        return self.decoder(z)

class VGAE(nn.Module):
    """
    Variational Graph Autoencoder for node generation
    """
    def __init__(self, encoder, decoder):
        super(VGAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, edge_index):
        """Forward pass: encode and sample"""
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode_nodes(self, z):
        """Decode latent vectors to node features"""
        return self.decoder(z)
    
    def encode(self, x, edge_index):
        """Encode nodes to latent space"""
        return self.encoder(x, edge_index)
    
    def decode_edges(self, z, edge_index):
        """Decode latent vectors to edge probabilities"""
        # Inner product decoder
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)
