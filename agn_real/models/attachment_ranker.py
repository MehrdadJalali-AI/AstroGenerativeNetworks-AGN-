from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from agn_real.utils import pyg_to_networkx, set_seed


@dataclass
class AttachmentConfig:
    hidden_dim: int = 64
    epochs: int = 30
    lr: float = 1e-3
    negative_ratio: int = 1
    seed: int = 0
    device: str = "cpu"


class AttachmentRanker(nn.Module):
    def __init__(self, pair_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features).view(-1)

    def predict_proba(self, pair_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(pair_features))


def node_roles(data: Data) -> Dict[str, torch.Tensor]:
    g = pyg_to_networkx(data)
    n = data.num_nodes
    deg = torch.tensor([g.degree(i) for i in range(n)], dtype=torch.float32)
    try:
        core_map = nx.core_number(g)
        core = torch.tensor([core_map.get(i, 0) for i in range(n)], dtype=torch.float32)
    except Exception:
        core = torch.zeros(n)
    try:
        if n > 10000:
            raise RuntimeError("skip pagerank for large graph")
        pr_map = nx.pagerank(g, max_iter=100)
        pr = torch.tensor([pr_map.get(i, 0.0) for i in range(n)], dtype=torch.float32)
    except Exception:
        pr = torch.zeros(n)
    try:
        if n > 1000:
            comms = list(nx.community.asyn_lpa_communities(g, seed=0))
        else:
            comms = list(nx.community.greedy_modularity_communities(g))
        comm = torch.zeros(n, dtype=torch.float32)
        for idx, nodes in enumerate(comms):
            for node in nodes:
                comm[int(node)] = float(idx)
    except Exception:
        comm = torch.zeros(n)
    return {"degree": deg, "core": core, "pagerank": pr, "community": comm}


def build_pair_features(
    node_x: torch.Tensor,
    node_z: Optional[torch.Tensor],
    roles: Dict[str, torch.Tensor],
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    x_sim = F.cosine_similarity(node_x[src], node_x[dst]).unsqueeze(1)
    if node_z is None:
        z_sim = x_sim
    else:
        z_sim = F.cosine_similarity(node_z[src], node_z[dst]).unsqueeze(1)
    deg_diff = (roles["degree"][src] - roles["degree"][dst]).abs().unsqueeze(1)
    deg_scale = roles["degree"].max().clamp_min(1.0)
    deg_diff = deg_diff / deg_scale
    core_match = (roles["core"][src].round() == roles["core"][dst].round()).float().unsqueeze(1)
    pr_diff = (roles["pagerank"][src] - roles["pagerank"][dst]).abs().unsqueeze(1)
    comm_match = (roles["community"][src].round() == roles["community"][dst].round()).float().unsqueeze(1)
    return torch.cat([z_sim, x_sim, deg_diff, core_match, pr_diff, comm_match], dim=1).float()


def train_attachment_ranker(data: Data, node_z: Optional[torch.Tensor] = None, config: AttachmentConfig | Dict[str, object] | None = None) -> Tuple[AttachmentRanker, Dict[str, object]]:
    if config is None:
        config = AttachmentConfig()
    if isinstance(config, dict):
        config = AttachmentConfig(**config)
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    node_x = data.x.float().to(device)
    node_z = node_z.float().to(device) if node_z is not None else None
    edge_index = data.edge_index.to(device)
    roles = {k: v.to(device) for k, v in node_roles(data).items()}
    neg = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=edge_index.size(1) * config.negative_ratio, method="sparse")
    src = torch.cat([edge_index[0], neg[0]])
    dst = torch.cat([edge_index[1], neg[1]])
    y = torch.cat([torch.ones(edge_index.size(1), device=device), torch.zeros(neg.size(1), device=device)])
    pairs = build_pair_features(node_x, node_z, roles, src, dst)
    model = AttachmentRanker(pairs.size(1), config.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(pairs)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        history.append({"epoch": epoch, "loss": float(loss.detach())})
    with torch.no_grad():
        probs = model.predict_proba(pairs).detach().cpu().numpy()
    labels = y.detach().cpu().numpy()
    try:
        auc = float(roc_auc_score(labels, probs))
        ap = float(average_precision_score(labels, probs))
    except Exception:
        auc, ap = float("nan"), float("nan")
    return model, {"history": history, "train_auc": auc, "train_ap": ap, "pair_dim": pairs.size(1)}


@torch.no_grad()
def attach_generated_nodes(
    observed_data: Data,
    generated_x: torch.Tensor,
    generated_z: Optional[torch.Tensor] = None,
    ranker: Optional[AttachmentRanker] = None,
    mode: str = "learned_ranker",
    k: int = 10,
    threshold: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Return edges connecting generated nodes to observed nodes.

    Generated node ids are offset after observed nodes.
    """
    rng = np.random.default_rng(seed)
    n_obs = observed_data.num_nodes
    n_gen = generated_x.size(0)
    obs_x = observed_data.x.float()
    all_x = torch.cat([obs_x, generated_x.float()], dim=0)
    all_z = None
    if generated_z is not None:
        obs_z = torch.zeros((n_obs, generated_z.size(1)), dtype=torch.float32)
        all_z = torch.cat([obs_z, generated_z.float()], dim=0)
    roles = node_roles(observed_data)
    for key in roles:
        pad = torch.zeros(n_gen, dtype=roles[key].dtype)
        roles[key] = torch.cat([roles[key], pad], dim=0)
    edges = []
    if mode == "random":
        for gi in range(n_gen):
            targets = rng.choice(n_obs, size=min(k, n_obs), replace=False)
            edges.extend([(n_obs + gi, int(t)) for t in targets])
    elif mode == "preferential":
        g = pyg_to_networkx(observed_data)
        deg = np.array([g.degree(i) + 1 for i in range(n_obs)], dtype=float)
        probs = deg / deg.sum()
        for gi in range(n_gen):
            targets = rng.choice(n_obs, size=min(k, n_obs), replace=False, p=probs)
            edges.extend([(n_obs + gi, int(t)) for t in targets])
    elif mode in {"cosine_topk", "community_aware"}:
        sims = cosine_similarity(generated_x.detach().cpu().numpy(), obs_x.detach().cpu().numpy())
        for gi in range(n_gen):
            targets = np.argsort(sims[gi])[-min(k, n_obs):]
            edges.extend([(n_obs + gi, int(t)) for t in targets if sims[gi, t] >= threshold or len(targets) <= k])
    elif mode == "learned_ranker":
        if ranker is None:
            raise ValueError("mode=learned_ranker requires a trained ranker")
        ranker_device = next(ranker.parameters()).device
        ranker.eval()
        for gi in range(n_gen):
            src = torch.full((n_obs,), n_obs + gi, dtype=torch.long)
            dst = torch.arange(n_obs, dtype=torch.long)
            pair = build_pair_features(all_x, all_z, roles, src, dst).to(ranker_device)
            scores = ranker.predict_proba(pair).detach().cpu()
            top = torch.topk(scores, k=min(k, n_obs)).indices.tolist()
            edges.extend([(n_obs + gi, int(t)) for t in top if float(scores[t]) >= threshold or len(top) <= k])
    else:
        raise ValueError(f"Unknown attachment mode: {mode}")
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)
