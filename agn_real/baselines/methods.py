from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from agn_real.models.attachment_ranker import attach_generated_nodes, train_attachment_ranker
from agn_real.models.rc_agn import RCAGNConfig, generate_nodes, train_model
from agn_real.utils import apply_feature_mode
from .fukushima_yamanishi_gca import run_fukushima_yamanishi_gca


def available_baselines() -> Dict[str, str]:
    return {
        "random": "Random node attributes from observed empirical distribution plus random attachment.",
        "preferential": "Random observed-like node attributes plus preferential attachment.",
        "knn_raw": "Hidden-count node insertion using raw feature nearest-neighbor prototypes.",
        "knn_raw_plus_structural": "kNN prototype baseline with raw plus structural features.",
        "standard_agn": "VGAE-style AGN ablation with standard Gaussian prior and cosine top-k attachment.",
        "vgae": "Alias for standard_agn with standard prior.",
        "gae": "Deterministic adapter is documented as unavailable in this lightweight implementation.",
        "sage_attach": "GraphSAGE encoder with learned attachment ranker.",
        "fukushima_yamanishi_gca": "Fukushima-Yamanishi-style community latent augmentation baseline.",
        "node2vec": "Adapter placeholder; unavailable unless a project-specific node2vec dependency is added.",
        "gca": "Adapter placeholder; unavailable unless a GCA implementation is supplied.",
    }


def _sample_observed_features(observed_data: Data, num_nodes: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    idx = rng.choice(observed_data.num_nodes, size=num_nodes, replace=True)
    noise = torch.randn((num_nodes, observed_data.x.size(1))) * 0.01
    return observed_data.x[torch.tensor(idx)].float() + noise


def _knn_features(observed_data: Data, num_nodes: int) -> torch.Tensor:
    x = observed_data.x.detach().cpu().numpy()
    n_neighbors = min(5, len(x))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    _, indices = nn.kneighbors(x[: min(num_nodes, len(x))])
    rows = []
    for i in range(num_nodes):
        rows.append(x[indices[i % len(indices)]].mean(axis=0))
    return torch.tensor(np.vstack(rows), dtype=torch.float32)


def run_baseline(
    name: str,
    observed_data: Data,
    num_nodes: int,
    seed: int = 0,
    k: int = 10,
    threshold: float = 0.5,
    epochs: int = 30,
) -> Dict[str, object]:
    name = name.lower()
    if name in {"node2vec", "deepwalk", "gca", "gae"}:
        return {"status": "unavailable", "reason": f"{name} adapter is present but not run without an implementation/dependency.", "generated_x": None, "edge_index": None}
    if name == "fukushima_yamanishi_gca":
        return run_fukushima_yamanishi_gca(observed_data, num_nodes, seed=seed, latent_dim=32, k=k, threshold=threshold)
    if name == "random":
        x = _sample_observed_features(observed_data, num_nodes, seed)
        edges = attach_generated_nodes(observed_data, x, mode="random", k=k, threshold=threshold, seed=seed)
        return {"status": "ok", "generated_x": x, "edge_index": edges, "attachment_mode": "random"}
    if name == "preferential":
        x = _sample_observed_features(observed_data, num_nodes, seed)
        edges = attach_generated_nodes(observed_data, x, mode="preferential", k=k, threshold=threshold, seed=seed)
        return {"status": "ok", "generated_x": x, "edge_index": edges, "attachment_mode": "preferential"}
    if name in {"knn_raw", "knn_raw_plus_structural"}:
        mode = "raw" if name == "knn_raw" else "raw_plus_structural"
        data = apply_feature_mode(observed_data, mode)
        x = _knn_features(data, num_nodes)
        edges = attach_generated_nodes(data, x, mode="cosine_topk", k=k, threshold=threshold, seed=seed)
        return {"status": "ok", "generated_x": x, "edge_index": edges, "attachment_mode": "cosine_topk"}
    if name in {"unconditioned_cosine"}:
        name = "standard_agn"
    if name in {"standard_agn", "vgae", "sage_attach"}:
        encoder = "sage" if name == "sage_attach" else "gcn"
        conditional = False
        cfg = RCAGNConfig(input_dim=observed_data.x.size(1), encoder_type=encoder, conditional_prior=conditional, epochs=epochs, seed=seed)
        model, train_info = train_model(observed_data, cfg)
        gen = generate_nodes(model, num_nodes)
        if name == "sage_attach":
            ranker, ranker_info = train_attachment_ranker(observed_data, config={"epochs": max(5, epochs // 2), "seed": seed})
            edges = attach_generated_nodes(observed_data, gen["x"], gen["z"], ranker=ranker, mode="learned_ranker", k=k, threshold=threshold, seed=seed)
            attach_info = ranker_info
        else:
            edges = attach_generated_nodes(observed_data, gen["x"], gen["z"], mode="cosine_topk", k=k, threshold=threshold, seed=seed)
            attach_info = {"mode": "cosine_topk"}
        return {"status": "ok", "generated_x": gen["x"], "generated_z": gen["z"], "edge_index": edges, "train_info": train_info, "attachment_info": attach_info}
    raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(available_baselines())}")
