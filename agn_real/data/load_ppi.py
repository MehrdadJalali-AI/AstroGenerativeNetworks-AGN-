from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import json
import zipfile

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import PPI
from torch_geometric.utils import subgraph

from agn_real.utils import clean_edge_index


def _limit_nodes(data: Data, max_nodes: Optional[int]) -> Data:
    if max_nodes is None or data.num_nodes <= max_nodes:
        return data
    nodes = torch.arange(max_nodes, dtype=torch.long)
    edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    out = Data(x=data.x[nodes].clone(), edge_index=clean_edge_index(edge_index, max_nodes, undirected=True), num_nodes=max_nodes)
    if torch.is_tensor(getattr(data, "y", None)) and data.y.size(0) == data.num_nodes:
        out.y = data.y[nodes].clone()
    return out


def _disjoint_union(dataset, max_nodes: Optional[int] = None) -> Data:
    xs, ys, edges = [], [], []
    offset = 0
    for data in dataset:
        d = data.clone()
        if max_nodes is not None and offset + d.num_nodes > max_nodes:
            d = _limit_nodes(d, max_nodes - offset)
        xs.append(d.x.float())
        if torch.is_tensor(getattr(d, "y", None)):
            ys.append(d.y)
        edges.append(d.edge_index + offset)
        offset += d.num_nodes
        if max_nodes is not None and offset >= max_nodes:
            break
    out = Data(x=torch.cat(xs, dim=0), edge_index=clean_edge_index(torch.cat(edges, dim=1), offset, undirected=True), num_nodes=offset)
    if ys:
        out.y = torch.cat(ys, dim=0)
    return out


def _maybe_extract_ppi_zip(base: Path) -> None:
    expected = base / "train_graph.json"
    if expected.exists():
        return
    for zip_path in [base / "ppi.zip", base / "raw" / "ppi.zip"]:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(base)
            # Some archives contain a nested ppi/ folder.
            nested = base / "ppi"
            if not expected.exists() and nested.exists():
                for item in nested.iterdir():
                    target = base / item.name
                    if not target.exists():
                        item.replace(target)
            return


def _has_graphsage_ppi(base: Path) -> bool:
    return all((base / f"train_{suffix}").exists() for suffix in ["graph.json", "feats.npy", "labels.npy", "graph_id.npy"])


def _load_graphsage_ppi_split(base: Path, split: str = "train") -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with open(base / f"{split}_graph.json") as f:
        graph = json.load(f)
    feats = np.load(base / f"{split}_feats.npy")
    labels = np.load(base / f"{split}_labels.npy")
    graph_id = np.load(base / f"{split}_graph_id.npy")
    return graph, feats, labels, graph_id


def _graphsage_to_data(base: Path, mode: str, graph_index: int, max_nodes: Optional[int]) -> Tuple[Data, dict]:
    graph, feats, labels, graph_id = _load_graphsage_ppi_split(base, "train")
    node_ids = np.arange(feats.shape[0])
    mode = mode.lower()
    if mode == "selected_graph":
        graph_ids = sorted(np.unique(graph_id).tolist())
        if graph_index < 0 or graph_index >= len(graph_ids):
            raise ValueError(f"ppi_graph_index={graph_index} outside available range 0..{len(graph_ids)-1}")
        selected_gid = graph_ids[graph_index]
        selected = node_ids[graph_id == selected_gid]
        mode_note = f"selected_graph:{graph_index}:graph_id={selected_gid}"
    elif mode == "disjoint_union":
        selected = node_ids
        selected_gid = None
        mode_note = "disjoint_union:train"
    else:
        raise ValueError("ppi_mode must be selected_graph or disjoint_union")
    if max_nodes is not None:
        selected = selected[:max_nodes]
    selected_set = set(int(i) for i in selected.tolist())
    old_to_new = {int(old): i for i, old in enumerate(selected.tolist())}
    edges = []
    for link in graph.get("links", []):
        s = int(link["source"])
        t = int(link["target"])
        if s in selected_set and t in selected_set and s != t:
            edges.append((old_to_new[s], old_to_new[t]))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(
        x=torch.tensor(feats[selected], dtype=torch.float32),
        y=torch.tensor(labels[selected], dtype=torch.float32),
        edge_index=clean_edge_index(edge_index, len(selected), undirected=True),
        num_nodes=len(selected),
    )
    meta = {
        "dataset": "PPI",
        "domain": "biological/PPI",
        "source": "local GraphSAGE/DGL PPI",
        "ppi_root": str(base),
        "ppi_mode": mode_note,
        "ppi_num_graphs": int(len(np.unique(graph_id))),
        "ppi_graph_index": graph_index if mode == "selected_graph" else None,
        "ppi_graph_id": int(selected_gid) if selected_gid is not None else None,
        "ppi_max_nodes": max_nodes,
    }
    return data, meta


def load_ppi_dataset(
    root: str | Path,
    ppi_root: str | Path | None = None,
    mode: str = "selected_graph",
    graph_index: int = 0,
    max_nodes: int | None = None,
    allow_download: bool = False,
) -> Tuple[Data, dict]:
    """Load PyG PPI from cache/download or a caller-supplied local root.

    `ppi_root` may point either to the PyG PPI root containing `raw/` and
    `processed/`, or to a parent directory where those files should be created.
    """
    base = Path(ppi_root) if ppi_root else Path(root) / "PPI"
    candidates = [base, Path(root) / "PPI", Path(root) / "ppi", Path(root) / "real" / "PPI", Path(root) / "real" / "ppi"]
    for candidate in candidates:
        _maybe_extract_ppi_zip(candidate)
        if _has_graphsage_ppi(candidate):
            base = candidate
            break
    if _has_graphsage_ppi(base):
        return _graphsage_to_data(base, mode=mode, graph_index=graph_index, max_nodes=max_nodes)
    processed = base / "processed"
    raw = base / "raw"
    has_local = (processed.exists() and any(processed.glob("*.pt"))) or (raw.exists() and any(raw.iterdir()))
    if not allow_download and not has_local:
        raise FileNotFoundError(
            f"No local PyG PPI cache found under {base}. Provide --ppi_root pointing to a local PPI root "
            "with raw/ or processed/ files, or explicitly enable online download in code."
        )
    dataset = PPI(root=str(base), split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"No PPI graphs found under {base}")
    mode = mode.lower()
    if mode == "selected_graph":
        if graph_index < 0 or graph_index >= len(dataset):
            raise ValueError(f"ppi_graph_index={graph_index} outside available range 0..{len(dataset)-1}")
        data = dataset[graph_index].clone()
        mode_note = f"selected_graph:{graph_index}"
    elif mode == "disjoint_union":
        data = _disjoint_union(dataset, max_nodes=max_nodes)
        mode_note = "disjoint_union"
        max_nodes = None
    else:
        raise ValueError("ppi_mode must be selected_graph or disjoint_union")
    data.edge_index = clean_edge_index(data.edge_index, data.num_nodes, undirected=True)
    data.x = data.x.float()
    data = _limit_nodes(data, max_nodes)
    meta = {
        "dataset": "PPI",
        "domain": "biological/PPI",
        "source": "PyG PPI",
        "ppi_root": str(base),
        "ppi_mode": mode_note,
        "ppi_num_graphs": len(dataset),
        "ppi_graph_index": graph_index if mode == "selected_graph" else None,
        "ppi_max_nodes": max_nodes,
    }
    return data, meta
