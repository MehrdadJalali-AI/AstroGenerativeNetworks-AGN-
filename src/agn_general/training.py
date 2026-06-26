"""
Training for AGN: edge reconstruction + feature reconstruction + KL, with validation early stopping.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected

from .config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_DIR,
    BETA,
    GAMMA,
    EARLY_STOPPING_PATIENCE,
    USE_VALIDATION_EARLY_STOPPING,
    LINK_SPLIT_NUM_VAL,
    LINK_SPLIT_NUM_TEST,
)


def edge_reconstruction_loss(pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
    pos_loss = -torch.log(pos_pred + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
    return pos_loss + neg_loss


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def feature_reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE on normalized features (per manuscript)."""
    return F.mse_loss(x_hat, x)


def forward_losses(
    model: torch.nn.Module,
    data: Data,
    beta: float,
    gamma: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    z, mu, logvar = model(data.x, data.edge_index)
    pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
    neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
    pos_pred = model.decode_edges(z, pos_edge_index)
    neg_pred = model.decode_edges(z, neg_edge_index)
    l_recon = edge_reconstruction_loss(pos_pred, neg_pred)
    x_hat = model.decode_nodes(z)
    l_feat = feature_reconstruction_loss(x_hat, data.x)
    l_kl = kl_loss(mu, logvar)
    total = l_recon + gamma * l_feat + beta * l_kl
    parts = {
        "l_recon": float(l_recon.detach()),
        "l_feat": float(l_feat.detach()),
        "l_kl": float(l_kl.detach()),
        "total": float(total.detach()),
    }
    return total, parts


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    beta: float,
    gamma: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    d = data.to(device)
    total, parts = forward_losses(model, d, beta=beta, gamma=gamma)
    return float(total.detach()), parts


def train_epoch(
    model: torch.nn.Module,
    data: Data,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
    gamma: float,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    d = data.to(device)
    total, parts = forward_losses(model, d, beta=beta, gamma=gamma)
    total.backward()
    optimizer.step()
    return parts


def run_training(
    model: torch.nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    beta: float = BETA,
    gamma: float = GAMMA,
    patience: int = EARLY_STOPPING_PATIENCE,
    use_validation_early_stopping: bool = USE_VALIDATION_EARLY_STOPPING,
    model_dir: Optional[str] = None,
    checkpoint_name: str = "best_agn_model.pth",
) -> Dict[str, Any]:
    """
    Train VGAE with L = L_recon + gamma * L_feat + beta * L_KL.
    Best checkpoint is chosen by validation total loss when use_validation_early_stopping else training loss.
    """
    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, checkpoint_name)

    data = Data(x=features, edge_index=to_undirected(edge_index))
    splitter = RandomLinkSplit(
        num_val=LINK_SPLIT_NUM_VAL,
        num_test=LINK_SPLIT_NUM_TEST,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = splitter(data)

    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_metric = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {epochs} epochs (beta={beta}, gamma={gamma})...")
    print(f"Device: {DEVICE} | Nodes: {features.shape[0]} | Features: {features.shape[1]}")
    print(f"Validation early stopping: {use_validation_early_stopping}")

    for epoch in range(1, epochs + 1):
        parts = train_epoch(model, train_data, optimizer, DEVICE, beta, gamma)
        train_losses.append(parts["total"])

        if use_validation_early_stopping:
            val_total, _ = evaluate_split(model, val_data, DEVICE, beta, gamma)
            val_losses.append(val_total)
            monitor = val_total
        else:
            val_losses.append(float("nan"))
            monitor = parts["total"]

        if epoch % 10 == 0:
            msg = f"Epoch {epoch}/{epochs} | train {parts['total']:.4f} (recon {parts['l_recon']:.4f}, feat {parts['l_feat']:.4f}, kl {parts['l_kl']:.4f})"
            if use_validation_early_stopping:
                msg += f" | val {monitor:.4f}"
            print(msg)

        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best monitor loss {best_metric:.4f})")
                break

    if os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded best checkpoint from {ckpt_path} (epoch {best_epoch})")
    else:
        print("Warning: no checkpoint saved")

    test_total, test_parts = evaluate_split(model, test_data.to(DEVICE), DEVICE, beta, gamma)
    print(f"Held-out test loss: {test_total:.4f} (recon {test_parts['l_recon']:.4f}, feat {test_parts['l_feat']:.4f}, kl {test_parts['l_kl']:.4f})")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_or_train_loss": best_metric,
        "best_epoch": best_epoch,
        "test_loss": test_total,
        "test_parts": test_parts,
        "checkpoint_path": ckpt_path,
    }
