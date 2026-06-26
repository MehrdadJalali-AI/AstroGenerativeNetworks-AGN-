from __future__ import annotations

import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_networkx, to_undirected


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def pyg_to_networkx(data: Data, undirected: bool = True) -> nx.Graph:
    return to_networkx(data, to_undirected=undirected, remove_self_loops=True)


def clean_edge_index(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True) -> torch.Tensor:
    edge_index, _ = remove_self_loops(edge_index)
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index.long()


def relabel_subgraph(data: Data, keep_nodes: Sequence[int]) -> Tuple[Data, Dict[int, int], Dict[int, int]]:
    keep = torch.as_tensor(sorted(int(n) for n in keep_nodes), dtype=torch.long)
    old_to_new = {int(old): i for i, old in enumerate(keep.tolist())}
    new_to_old = {i: int(old) for i, old in enumerate(keep.tolist())}
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[keep] = True
    src, dst = data.edge_index
    edge_mask = mask[src] & mask[dst]
    old_edges = data.edge_index[:, edge_mask]
    relabeled = torch.empty_like(old_edges)
    for row in (0, 1):
        relabeled[row] = torch.as_tensor([old_to_new[int(v)] for v in old_edges[row].tolist()], dtype=torch.long)
    out = Data(edge_index=clean_edge_index(relabeled, len(keep), undirected=True), num_nodes=len(keep))
    for key, value in data:
        if key in {"edge_index", "num_nodes"}:
            continue
        if torch.is_tensor(value) and value.size(0) == data.num_nodes:
            setattr(out, key, value[keep].clone())
        else:
            setattr(out, key, value)
    return out, old_to_new, new_to_old


def structural_features(data: Data) -> torch.Tensor:
    g = pyg_to_networkx(data)
    n = data.num_nodes
    degree = np.array([g.degree(i) for i in range(n)], dtype=np.float32)
    clustering = np.array(list(nx.clustering(g, nodes=range(n)).values()), dtype=np.float32)
    try:
        core_map = nx.core_number(g)
        core = np.array([core_map.get(i, 0) for i in range(n)], dtype=np.float32)
    except Exception:
        core = np.zeros(n, dtype=np.float32)
    try:
        if n > 50000:
            raise RuntimeError("skip exact pagerank for very large graph")
        pr_map = nx.pagerank(g, max_iter=100)
        pagerank = np.array([pr_map.get(i, 0.0) for i in range(n)], dtype=np.float32)
    except Exception:
        pagerank = np.zeros(n, dtype=np.float32)
    features = np.stack([degree, clustering, core, pagerank], axis=1)
    maxes = np.nanmax(np.abs(features), axis=0)
    maxes[maxes == 0] = 1.0
    return torch.tensor(features / maxes, dtype=torch.float32)


def apply_feature_mode(data: Data, feature_mode: str) -> Data:
    out = data.clone()
    raw = out.x.float() if getattr(out, "x", None) is not None else torch.empty((out.num_nodes, 0))
    struct = structural_features(out)
    if feature_mode == "raw":
        if raw.numel() == 0:
            raise ValueError("feature_mode=raw requested, but dataset has no raw node attributes")
        out.x = raw
    elif feature_mode == "raw_plus_structural":
        if raw.numel() == 0:
            out.x = struct
        else:
            out.x = torch.cat([raw, struct], dim=1)
    elif feature_mode == "structural_only":
        out.x = struct
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    return out


def record_from_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in mapping.items():
        if is_dataclass(value):
            out[key] = asdict(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif torch.is_tensor(value):
            out[key] = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = str(value)
    return out
