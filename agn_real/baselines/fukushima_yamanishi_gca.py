from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from agn_real.models.attachment_ranker import attach_generated_nodes
from agn_real.utils import pyg_to_networkx, set_seed


def _communities(data: Data, seed: int) -> List[List[int]]:
    g = pyg_to_networkx(data)
    try:
        comms = nx.community.louvain_communities(g, seed=seed)
    except Exception:
        comms = nx.community.greedy_modularity_communities(g)
    return [sorted(int(n) for n in c) for c in comms if c]


def _latent(data: Data, latent_dim: int, seed: int) -> np.ndarray:
    x = data.x.detach().cpu().numpy()
    n_components = min(latent_dim, max(1, min(x.shape) - 1))
    x_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    if x.shape[1] > 200:
        return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(x_scaled)
    return PCA(n_components=n_components, random_state=seed).fit_transform(x_scaled)


def run_fukushima_yamanishi_gca(
    observed_data: Data,
    num_nodes: int,
    seed: int = 0,
    latent_dim: int = 32,
    k: int = 10,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Fukushima-Yamanishi-style community latent augmentation baseline.

    This is a faithful approximation from the manuscript-level description when
    exact ICDM 2024 code/details are unavailable: detect communities, learn a
    latent node representation, fit community-conditioned Gaussian mixtures,
    sample latent points, decode them back to feature space with a linear map,
    and attach generated nodes with community-aware similarity rules.

    It is intentionally labeled as style/approximation, not exact reproduction.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    z = _latent(observed_data, latent_dim, seed)
    x = observed_data.x.detach().cpu().numpy()
    comms = _communities(observed_data, seed)
    if not comms:
        return {"status": "unavailable", "reason": "community detection produced no communities"}

    community_sizes = np.array([len(c) for c in comms], dtype=float)
    community_probs = community_sizes / community_sizes.sum()
    decoder = np.linalg.lstsq(np.c_[z, np.ones(z.shape[0])], x, rcond=None)[0]
    sampled_z = []
    sampled_communities = []
    for _ in range(num_nodes):
        ci = int(rng.choice(len(comms), p=community_probs))
        nodes = comms[ci]
        local_z = z[nodes]
        n_components = min(3, max(1, len(nodes) // 5), len(nodes))
        if len(nodes) < 2:
            sample = local_z[0:1]
        else:
            gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=seed, reg_covar=1e-4)
            gmm.fit(local_z)
            sample, _ = gmm.sample(1)
        sampled_z.append(sample.reshape(-1))
        sampled_communities.append(ci)
    gen_z_np = np.vstack(sampled_z)
    gen_x_np = np.c_[gen_z_np, np.ones(gen_z_np.shape[0])] @ decoder
    gen_x = torch.tensor(gen_x_np, dtype=torch.float32)
    gen_z = torch.tensor(gen_z_np, dtype=torch.float32)
    edges = attach_generated_nodes(
        observed_data,
        gen_x,
        gen_z,
        mode="community_aware",
        k=k,
        threshold=threshold,
        seed=seed,
    )
    return {
        "status": "ok",
        "generated_x": gen_x,
        "generated_z": gen_z,
        "edge_index": edges,
        "baseline_label": "Fukushima-Yamanishi-style community latent augmentation baseline",
        "sampled_communities": sampled_communities,
    }
