from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv

from agn_real.utils import set_seed


@dataclass
class ClassifierConfig:
    hidden_dim: int = 64
    epochs: int = 50
    lr: float = 1e-2
    weight_decay: float = 5e-4
    encoder: str = "gcn"
    seed: int = 0
    device: str = "cpu"


class NodeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, encoder: str = "gcn", multilabel: bool = False):
        super().__init__()
        conv = SAGEConv if encoder == "sage" else GCNConv
        self.conv1 = conv(input_dim, hidden_dim)
        self.conv2 = conv(hidden_dim, output_dim)
        self.multilabel = multilabel

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.35, training=self.training)
        return self.conv2(h, edge_index)


def prune_generated_edges(
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor | None,
    observed_num_nodes: int,
    threshold: float | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    if edge_index.numel() == 0 or (threshold is None and top_k is None):
        return edge_index
    if edge_scores is None or edge_scores.numel() != edge_index.size(1):
        edge_scores = torch.ones(edge_index.size(1), dtype=torch.float32)
    scores = edge_scores.detach().cpu().float()
    keep = torch.ones(edge_index.size(1), dtype=torch.bool)
    generated_mask = (edge_index[0].cpu() >= observed_num_nodes) | (edge_index[1].cpu() >= observed_num_nodes)
    if threshold is not None:
        keep &= (~generated_mask) | (scores >= float(threshold))
    if top_k is not None and top_k > 0:
        gen_idx = torch.where(generated_mask & keep)[0]
        if gen_idx.numel() > top_k:
            top = gen_idx[torch.topk(scores[gen_idx], k=top_k).indices]
            top_mask = torch.zeros_like(keep)
            top_mask[top] = True
            keep &= (~generated_mask) | top_mask
    return edge_index[:, keep.to(edge_index.device)]


def dropout_generated_edges(edge_index: torch.Tensor, observed_num_nodes: int, p: float, seed: int = 0) -> torch.Tensor:
    if edge_index.numel() == 0 or p <= 0:
        return edge_index
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    generated_mask = (edge_index[0].cpu() >= observed_num_nodes) | (edge_index[1].cpu() >= observed_num_nodes)
    random_keep = torch.rand(edge_index.size(1), generator=gen) >= float(p)
    keep = (~generated_mask) | random_keep
    return edge_index[:, keep.to(edge_index.device)]


def random_label_split(num_nodes: int, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_nodes)
    n_train = max(1, int(train_ratio * num_nodes))
    n_val = max(1, int(val_ratio * num_nodes))
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    train[idx[:n_train]] = True
    val[idx[n_train : n_train + n_val]] = True
    test[idx[n_train + n_val :]] = True
    if test.sum() == 0:
        test[idx[-1]] = True
    return train, val, test


def _metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, multilabel: bool) -> Dict[str, float]:
    if mask.sum() == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "micro_f1": float("nan")}
    if multilabel:
        prob = torch.sigmoid(logits[mask]).cpu().numpy()
        pred = (prob >= 0.5).astype(int)
        true = y[mask].int().cpu().numpy()
        try:
            micro_auroc = float(roc_auc_score(true, prob, average="micro"))
        except Exception:
            micro_auroc = float("nan")
        try:
            macro_auroc = float(roc_auc_score(true, prob, average="macro"))
        except Exception:
            macro_auroc = float("nan")
        return {
            "accuracy": float("nan"),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(true, pred, average="micro", zero_division=0)),
            "macro_auroc": macro_auroc,
            "micro_auroc": micro_auroc,
        }
    pred = logits[mask].argmax(dim=1).cpu().numpy()
    true = y[mask].view(-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true, pred, average="micro", zero_division=0)),
        "macro_auroc": float("nan"),
        "micro_auroc": float("nan"),
    }


def train_and_evaluate_node_classifier(
    data: Data,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    config: ClassifierConfig,
) -> Dict[str, float]:
    if not torch.is_tensor(getattr(data, "y", None)) or data.y.numel() == 0:
        return {"status": "not_run", "reason": "dataset has no labels"}
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    d = data.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    y = d.y.float() if d.y.dim() > 1 and d.y.size(1) > 1 else d.y.long().view(-1)
    multilabel = y.dim() > 1 and y.size(1) > 1
    output_dim = y.size(1) if multilabel else int(torch.unique(y[train_mask]).numel() if torch.unique(y[train_mask]).numel() > 0 else torch.unique(y).numel())
    if not multilabel:
        output_dim = int(torch.max(y).item()) + 1
    model = NodeClassifier(d.x.size(1), config.hidden_dim, output_dim, config.encoder, multilabel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for _ in range(config.epochs):
        model.train()
        opt.zero_grad()
        logits = model(d.x.float(), d.edge_index)
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        else:
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(d.x.float(), d.edge_index)
    out = _metrics(logits, y, test_mask, multilabel)
    out["status"] = "ok"
    return out
