from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents, NormalizeFeatures, ToUndirected
from torch_geometric.utils import is_undirected

from agn_real.utils import clean_edge_index, ensure_dir, pyg_to_networkx
from agn_real.data.load_ppi import load_ppi_dataset


_PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
_COAUTHOR = {"coauthorcs": "CS", "coauthor_cs": "CS", "coauthorphysics": "Physics", "coauthor_physics": "Physics"}
_AMAZON = {"amazoncomputers": "Computers", "amazon_computers": "Computers", "amazonphoto": "Photo", "amazon_photo": "Photo"}
_DOMAIN = {
    "cora": "citation",
    "citeseer": "citation",
    "pubmed": "citation",
    "ogbnarxiv": "citation",
    "ogbn_arxiv": "citation",
    "ogb_arxiv": "citation",
    "ppi": "biological/PPI",
    "coauthorcs": "coauthorship",
    "coauthor_cs": "coauthorship",
    "coauthorphysics": "coauthorship",
    "coauthor_physics": "coauthorship",
    "amazoncomputers": "product co-purchase",
    "amazon_computers": "product co-purchase",
    "amazonphoto": "product co-purchase",
    "amazon_photo": "product co-purchase",
}


def _copy_data(data: Data) -> Data:
    copied = data.clone()
    copied.edge_index = clean_edge_index(copied.edge_index, copied.num_nodes, undirected=True)
    if getattr(copied, "x", None) is not None:
        copied.x = copied.x.float()
    return copied


def _load_dataset_object(name: str, root: str | Path):
    key = name.lower().replace("-", "_").replace(" ", "")
    root = Path(root)
    if key in _PLANETOID:
        from torch_geometric.datasets import Planetoid

        return Planetoid(root=str(root / "Planetoid"), name=_PLANETOID[key])
    if key in _COAUTHOR:
        from torch_geometric.datasets import Coauthor

        return Coauthor(root=str(root / "Coauthor"), name=_COAUTHOR[key])
    if key in _AMAZON:
        from torch_geometric.datasets import Amazon

        return Amazon(root=str(root / "Amazon"), name=_AMAZON[key])
    if key == "ppi":
        raise ValueError("PPI is handled by load_ppi_dataset")
    if key in {"ogbnarxiv", "ogbn_arxiv", "ogb_arxiv"}:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as exc:
            raise ImportError("OGBN-Arxiv requires the optional ogb package.") from exc
        return PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "OGB"))
    raise ValueError(
        f"Unknown real dataset '{name}'. Supported: Cora, CiteSeer, PubMed, "
        "CoauthorCS, CoauthorPhysics, AmazonComputers, AmazonPhoto, optional OGBN-Arxiv."
    )


def load_real_dataset(
    name: str,
    root: str | Path,
    make_undirected: bool = True,
    largest_component: bool = True,
    normalize_features: bool = True,
    ppi_root: str | Path | None = None,
    ppi_mode: str = "selected_graph",
    ppi_graph_index: int = 0,
    ppi_max_nodes: int | None = None,
    ppi_allow_download: bool = False,
) -> Tuple[Data, Dict[str, object]]:
    """Load a real PyG graph and return `(data, metadata)`.

    The function preserves raw node attributes where available, optionally normalizes
    them with PyG's row-normalization transform, and removes all but the largest
    connected component by default to stabilize topology metrics.
    """
    key = name.lower().replace("-", "_").replace(" ", "")
    if key == "ppi":
        data, ppi_meta = load_ppi_dataset(
            root=root,
            ppi_root=ppi_root,
            mode=ppi_mode,
            graph_index=ppi_graph_index,
            max_nodes=ppi_max_nodes,
            allow_download=ppi_allow_download,
        )
        dataset = None
    else:
        dataset = _load_dataset_object(name, root)
        data = _copy_data(dataset[0])
    if make_undirected:
        data = ToUndirected()(data)
        data.edge_index = clean_edge_index(data.edge_index, data.num_nodes, undirected=True)
    if normalize_features and getattr(data, "x", None) is not None:
        data = NormalizeFeatures()(data)
        data.x = data.x.float()
    if largest_component:
        data = LargestConnectedComponents(num_components=1)(data)
        data.edge_index = clean_edge_index(data.edge_index, data.num_nodes, undirected=True)
    metadata = summarize_dataset(data)
    metadata["dataset"] = name
    metadata["domain"] = _DOMAIN.get(name.lower().replace("-", "_").replace(" ", ""), "real network")
    metadata["source"] = dataset.__class__.__name__ if dataset is not None else ppi_meta.get("source", "PyG PPI")
    if key == "ppi":
        metadata.update(ppi_meta)
    metadata["has_raw_features"] = getattr(data, "x", None) is not None
    return data, metadata


def summarize_dataset(data: Data) -> Dict[str, object]:
    g = pyg_to_networkx(data, undirected=True)
    n = int(data.num_nodes)
    e = int(g.number_of_edges())
    y = getattr(data, "y", None)
    if torch.is_tensor(y) and y.numel():
        if y.dim() > 1 and y.size(1) > 1:
            num_classes = int(y.size(1))
        else:
            num_classes = int(torch.unique(y).numel())
    else:
        num_classes = None
    density = nx.density(g) if n > 1 else 0.0
    degrees = np.array([deg for _, deg in g.degree()], dtype=float)
    try:
        if n > 5000:
            clustering = nx.algorithms.approximation.average_clustering(g, trials=min(1000, n), seed=0)
        else:
            clustering = nx.average_clustering(g)
    except Exception:
        clustering = float("nan")
    return {
        "num_nodes": n,
        "num_edges": e,
        "feature_dim": int(data.x.size(1)) if getattr(data, "x", None) is not None else 0,
        "num_classes": num_classes,
        "is_undirected": bool(is_undirected(data.edge_index, num_nodes=n)),
        "density": float(density),
        "average_degree": float(degrees.mean()) if degrees.size else 0.0,
        "clustering_approx": float(clustering),
    }


def save_dataset_summary_csv(summaries: Iterable[Dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(list(summaries)).to_csv(path, index=False)
    return path
