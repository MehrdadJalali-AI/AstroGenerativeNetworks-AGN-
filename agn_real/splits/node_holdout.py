from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from agn_real.utils import pyg_to_networkx, relabel_subgraph, set_seed


@dataclass
class NodeHoldoutSplit:
    observed_data: Data
    hidden_node_ids: torch.Tensor
    hidden_edges: torch.Tensor
    full_data_reference: Data
    old_to_observed: Dict[int, int]
    observed_to_old: Dict[int, int]
    strategy: str
    seed: int
    holdout_ratio: float


def _sample_by_bins(values: np.ndarray, n_holdout: int, rng: np.random.Generator, bins: int = 10) -> List[int]:
    if len(values) == 0:
        return []
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, min(bins, len(values)) + 1)))
    if len(quantiles) <= 2:
        return rng.choice(len(values), size=n_holdout, replace=False).tolist()
    bucket_ids = np.digitize(values, quantiles[1:-1], right=True)
    selected: List[int] = []
    for bucket in np.unique(bucket_ids):
        idx = np.where(bucket_ids == bucket)[0]
        target = int(round(n_holdout * len(idx) / len(values)))
        if target > 0:
            selected.extend(rng.choice(idx, size=min(target, len(idx)), replace=False).tolist())
    if len(selected) < n_holdout:
        remaining = np.array([i for i in range(len(values)) if i not in set(selected)])
        selected.extend(rng.choice(remaining, size=n_holdout - len(selected), replace=False).tolist())
    return selected[:n_holdout]


def _class_stratified(data: Data, n_holdout: int, rng: np.random.Generator) -> List[int]:
    if not torch.is_tensor(getattr(data, "y", None)) or data.y.numel() == 0:
        raise ValueError("class_stratified requires data.y labels")
    labels = data.y.detach().cpu().view(-1).numpy()
    selected: List[int] = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        target = int(round(n_holdout * len(idx) / len(labels)))
        if target > 0:
            selected.extend(rng.choice(idx, size=min(target, len(idx)), replace=False).tolist())
    if len(selected) < n_holdout:
        remaining = np.array([i for i in range(data.num_nodes) if i not in set(selected)])
        selected.extend(rng.choice(remaining, size=n_holdout - len(selected), replace=False).tolist())
    return selected[:n_holdout]


def _community_stratified(data: Data, n_holdout: int, rng: np.random.Generator) -> List[int]:
    g = pyg_to_networkx(data)
    try:
        communities = list(nx.community.louvain_communities(g, seed=int(rng.integers(0, 2**32 - 1))))
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(g))
    selected: List[int] = []
    total = max(1, data.num_nodes)
    for community in communities:
        nodes = np.array(sorted(community), dtype=int)
        target = int(round(n_holdout * len(nodes) / total))
        if target > 0 and len(nodes) > 0:
            selected.extend(rng.choice(nodes, size=min(target, len(nodes)), replace=False).tolist())
    if len(selected) < n_holdout:
        remaining = np.array([i for i in range(data.num_nodes) if i not in set(selected)])
        selected.extend(rng.choice(remaining, size=n_holdout - len(selected), replace=False).tolist())
    return selected[:n_holdout]


def stratified_node_holdout(data: Data, holdout_ratio: float, seed: int, strategy: str = "degree_stratified") -> NodeHoldoutSplit:
    """Hide real nodes and all incident edges, then relabel the observed graph.

    The returned `observed_data` contains only nodes not selected for holdout and
    only edges whose two endpoints are observed.
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_holdout = max(1, int(round(data.num_nodes * holdout_ratio)))
    strategy = strategy.lower()
    if strategy == "random":
        hidden = rng.choice(data.num_nodes, size=n_holdout, replace=False).tolist()
    elif strategy == "degree_stratified":
        g = pyg_to_networkx(data)
        degrees = np.array([g.degree(i) for i in range(data.num_nodes)], dtype=float)
        hidden = _sample_by_bins(degrees, n_holdout, rng)
    elif strategy == "class_stratified":
        hidden = _class_stratified(data, n_holdout, rng)
    elif strategy == "community_stratified":
        hidden = _community_stratified(data, n_holdout, rng)
    else:
        raise ValueError("strategy must be random, degree_stratified, class_stratified, or community_stratified")

    hidden_set = set(int(i) for i in hidden)
    observed_nodes = [i for i in range(data.num_nodes) if i not in hidden_set]
    src, dst = data.edge_index
    hidden_mask = torch.tensor([(int(s) in hidden_set) or (int(d) in hidden_set) for s, d in zip(src, dst)], dtype=torch.bool)
    hidden_edges = data.edge_index[:, hidden_mask].clone()
    observed_data, old_to_new, new_to_old = relabel_subgraph(data, observed_nodes)
    observed_data.hidden_old_node_ids = torch.tensor(sorted(hidden_set), dtype=torch.long)
    observed_data.observed_old_node_ids = torch.tensor(observed_nodes, dtype=torch.long)
    return NodeHoldoutSplit(
        observed_data=observed_data,
        hidden_node_ids=torch.tensor(sorted(hidden_set), dtype=torch.long),
        hidden_edges=hidden_edges,
        full_data_reference=data.clone(),
        old_to_observed=old_to_new,
        observed_to_old=new_to_old,
        strategy=strategy,
        seed=seed,
        holdout_ratio=holdout_ratio,
    )

