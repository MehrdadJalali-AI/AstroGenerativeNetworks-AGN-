"""
Training module for Generalized AGN
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from .config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_DIR
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

def loss_function(pos_pred, neg_pred, mu, logvar, beta=1.0):
    """
    VGAE loss function: reconstruction + KL divergence
    
    Args:
        pos_pred: Predicted probabilities for positive edges
        neg_pred: Predicted probabilities for negative edges
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    """
    # Reconstruction loss (binary cross-entropy)
    pos_loss = -torch.log(pos_pred + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
    recon_loss = pos_loss + neg_loss
    
    # KL divergence loss
    kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return recon_loss + beta * kl_div

def train_epoch(model, data, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    z, mu, logvar = model(data.x, data.edge_index)
    
    # Decode edges
    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
    
    pos_pred = model.decode_edges(z, pos_edge_index)
    neg_pred = model.decode_edges(z, neg_edge_index)
    
    # Compute loss
    loss = loss_function(pos_pred, neg_pred, mu, logvar)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def run_training(model, features, edge_index, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Main training loop
    
    Args:
        model: VGAE model
        features: Node features tensor
        edge_index: Edge indices tensor
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        losses: List of training losses
    """
    from torch_geometric.data import Data
    
    # Prepare data
    data = Data(x=features, edge_index=to_undirected(edge_index))
    
    # Split edges for training
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = splitter(data)
    
    # Move to device
    train_data = train_data.to(DEVICE)
    model = model.to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    losses = []
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Training on {DEVICE}")
    print(f"Nodes: {features.shape[0]}, Features: {features.shape[1]}")
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_data, optimizer, DEVICE)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_agn_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    return losses
