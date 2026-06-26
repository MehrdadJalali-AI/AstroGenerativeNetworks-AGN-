from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling

from agn_real.utils import ensure_dir, pyg_to_networkx, set_seed


@dataclass
class RCAGNConfig:
    input_dim: int
    hidden_dim: int = 64
    latent_dim: int = 32
    encoder_type: str = "gcn"
    num_layers: int = 2
    num_role_bins: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1e-2
    feature_weight: float = 1.0
    feature_mmd_weight: float = 0.0
    feature_centroid_weight: float = 0.0
    feature_mode: str = "decoder"
    interpolation_lambda: float = 0.5
    conditional_prior: bool = True
    seed: int = 0
    device: str = "cpu"


def _conv(kind: str, in_dim: int, out_dim: int):
    kind = kind.lower()
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    if kind == "gat":
        return GATConv(in_dim, out_dim, heads=1, concat=False)
    raise ValueError("encoder_type must be gcn, sage, or gat")


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, encoder_type: str, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(_conv(encoder_type, input_dim, hidden_dim))
        for _ in range(max(0, num_layers - 1)):
            self.convs.append(_conv(encoder_type, hidden_dim, hidden_dim))
        self.mu = _conv(encoder_type, hidden_dim, latent_dim)
        self.logvar = _conv(encoder_type, hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.mu(h, edge_index), self.logvar(h, edge_index)


class RoleConditionedAGN(nn.Module):
    def __init__(self, config: RCAGNConfig):
        super().__init__()
        self.config = config
        self.encoder = GraphEncoder(config.input_dim, config.hidden_dim, config.latent_dim, config.encoder_type, config.num_layers)
        self.role_mu = nn.Embedding(config.num_role_bins, config.latent_dim)
        self.role_logvar = nn.Embedding(config.num_role_bins, config.latent_dim)
        self.feature_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def encode(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(data.x, data.edge_index)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def conditional_parameters(self, roles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.role_mu(roles), self.role_logvar(roles).clamp(-6, 4)

    def decode_nodes(self, z: torch.Tensor) -> torch.Tensor:
        return self.feature_decoder(z)

    def decode_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(logits)

    def forward(self, data: Data, roles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode_nodes(z)
        return z, mu, logvar, x_hat


def compute_role_ids(data: Data, num_bins: int, include_labels: bool = False) -> torch.Tensor:
    g = pyg_to_networkx(data)
    n = data.num_nodes
    degree = np.array([g.degree(i) for i in range(n)], dtype=float)
    try:
        core_map = nx.core_number(g)
        core = np.array([core_map.get(i, 0) for i in range(n)], dtype=float)
    except Exception:
        core = np.zeros(n)
    try:
        if n > 10000:
            raise RuntimeError("skip pagerank for large graph")
        pr = nx.pagerank(g, max_iter=100)
        pagerank = np.array([pr.get(i, 0.0) for i in range(n)], dtype=float)
    except Exception:
        pagerank = np.zeros(n)

    def qbin(values: np.ndarray, bins: int) -> np.ndarray:
        if np.all(values == values[0]):
            return np.zeros_like(values, dtype=int)
        qs = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        return np.digitize(values, qs[1:-1], right=True)

    component_bins = 4
    role = qbin(degree, component_bins) + component_bins * qbin(core, component_bins) + component_bins * component_bins * qbin(pagerank, component_bins)
    if include_labels and torch.is_tensor(getattr(data, "y", None)):
        labels = data.y.detach().cpu().view(-1).numpy().astype(int)
        role = role + labels
    role = role % num_bins
    return torch.tensor(role, dtype=torch.long)


def _kl_conditional(mu: torch.Tensor, logvar: torch.Tensor, prior_mu: torch.Tensor, prior_logvar: torch.Tensor) -> torch.Tensor:
    var_ratio = torch.exp(logvar - prior_logvar)
    diff = (mu - prior_mu).pow(2) / torch.exp(prior_logvar)
    kl = 0.5 * (prior_logvar - logvar + var_ratio + diff - 1.0)
    return kl.sum(dim=1).mean()


def _edge_loss(model: RoleConditionedAGN, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    pos = edge_index
    neg = negative_sampling(edge_index, num_nodes=z.size(0), num_neg_samples=pos.size(1), method="sparse")
    pos_pred = model.decode_edges(z, pos)
    neg_pred = model.decode_edges(z, neg)
    return -torch.log(pos_pred + 1e-12).mean() - torch.log(1 - neg_pred + 1e-12).mean()


def train_model(observed_data: Data, config: RCAGNConfig | Dict[str, object]) -> Tuple[RoleConditionedAGN, Dict[str, object]]:
    if isinstance(config, dict):
        config = RCAGNConfig(input_dim=observed_data.x.size(1), **{k: v for k, v in config.items() if k != "input_dim"})
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    data = observed_data.to(device)
    roles = compute_role_ids(observed_data, config.num_role_bins).to(device)
    from agn_real.models.feature_generator import (
        centroid_regularization_loss,
        compute_role_feature_stats,
        feature_mode_uses_centroid,
        feature_mode_uses_mmd,
        rbf_mmd,
    )

    role_stats = compute_role_feature_stats(observed_data, roles.detach().cpu(), config.num_role_bins)
    model = RoleConditionedAGN(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        opt.zero_grad()
        z, mu, logvar, x_hat = model(data, roles)
        if config.conditional_prior:
            p_mu, p_logvar = model.conditional_parameters(roles)
        else:
            p_mu, p_logvar = torch.zeros_like(mu), torch.zeros_like(logvar)
        loss_edge = _edge_loss(model, z, data.edge_index)
        loss_feat = F.mse_loss(x_hat, data.x)
        loss_mmd = rbf_mmd(x_hat, data.x) if config.feature_mmd_weight > 0 or feature_mode_uses_mmd(config.feature_mode) else x_hat.new_tensor(0.0)
        loss_centroid = centroid_regularization_loss(x_hat, roles, role_stats) if config.feature_centroid_weight > 0 or feature_mode_uses_centroid(config.feature_mode) else x_hat.new_tensor(0.0)
        loss_kl = _kl_conditional(mu, logvar, p_mu, p_logvar)
        loss = loss_edge + config.feature_weight * loss_feat + config.feature_mmd_weight * loss_mmd + config.feature_centroid_weight * loss_centroid + config.beta * loss_kl
        loss.backward()
        opt.step()
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "edge_loss": float(loss_edge.detach()),
                "feature_loss": float(loss_feat.detach()),
                "feature_mmd_loss": float(loss_mmd.detach()),
                "feature_centroid_loss": float(loss_centroid.detach()),
                "kl_loss": float(loss_kl.detach()),
            }
        )
    return model, {"history": history, "roles": roles.detach().cpu(), "role_feature_stats": role_stats, "config": asdict(config)}


@torch.no_grad()
def generate_nodes(model: RoleConditionedAGN, num_nodes: int, role_distribution: Optional[torch.Tensor] = None, config: Optional[RCAGNConfig] = None) -> Dict[str, torch.Tensor]:
    model.eval()
    cfg = config or model.config
    device = next(model.parameters()).device
    if role_distribution is None:
        probs = torch.ones(cfg.num_role_bins, device=device) / cfg.num_role_bins
    else:
        probs = role_distribution.float().to(device)
        probs = probs / probs.sum().clamp_min(1e-12)
    roles = torch.multinomial(probs, num_samples=num_nodes, replacement=True)
    if cfg.conditional_prior:
        mu, logvar = model.conditional_parameters(roles)
    else:
        mu = torch.zeros(num_nodes, cfg.latent_dim, device=device)
        logvar = torch.zeros_like(mu)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    x = model.decode_nodes(z)
    return {"x": x.detach().cpu(), "z": z.detach().cpu(), "roles": roles.detach().cpu()}


def save_checkpoint(model: RoleConditionedAGN, path: str | Path, metadata: Optional[Dict[str, object]] = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config), "metadata": metadata or {}}, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> RoleConditionedAGN:
    ckpt = torch.load(path, map_location=map_location)
    cfg = RCAGNConfig(**ckpt["config"])
    model = RoleConditionedAGN(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model
