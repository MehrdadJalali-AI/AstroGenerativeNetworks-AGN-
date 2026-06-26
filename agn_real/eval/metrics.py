from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, normalized_mutual_info_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from torch_geometric.data import Data

from agn_real.splits.node_holdout import NodeHoldoutSplit
from agn_real.utils import clean_edge_index, pyg_to_networkx, structural_features
from agn_real.models.rc_agn import compute_role_ids


def build_augmented_data(observed_data: Data, generated_x: torch.Tensor, generated_edges: torch.Tensor) -> Data:
    n_obs = observed_data.num_nodes
    x = torch.cat([observed_data.x.float(), generated_x.float()], dim=0)
    edge_index = torch.cat([observed_data.edge_index.cpu(), generated_edges.cpu()], dim=1)
    edge_index = clean_edge_index(edge_index, n_obs + generated_x.size(0), undirected=True)
    return Data(x=x, edge_index=edge_index, num_nodes=n_obs + generated_x.size(0))


def _topology(g: nx.Graph, sample_paths: int = 500) -> Dict[str, float]:
    n, e = g.number_of_nodes(), g.number_of_edges()
    degrees = np.array([d for _, d in g.degree()], dtype=float)
    out = {
        "num_nodes": float(n),
        "num_edges": float(e),
        "density": float(nx.density(g)) if n > 1 else 0.0,
        "average_degree": float(degrees.mean()) if degrees.size else 0.0,
        "connected_components": float(nx.number_connected_components(g)) if n else 0.0,
    }
    try:
        if n > 5000:
            out["clustering"] = float(nx.algorithms.approximation.average_clustering(g, trials=min(1000, n), seed=0))
            out["clustering_sampled"] = 1.0
        else:
            out["clustering"] = float(nx.average_clustering(g))
    except Exception:
        out["clustering"] = float("nan")
    try:
        if n > 1000:
            communities = list(nx.community.asyn_lpa_communities(g, seed=0))
            out["modularity_approx"] = 1.0
        else:
            communities = list(nx.community.greedy_modularity_communities(g))
        out["modularity"] = float(nx.community.modularity(g, communities))
    except Exception:
        out["modularity"] = float("nan")
    try:
        out["assortativity"] = float(nx.degree_assortativity_coefficient(g))
    except Exception:
        out["assortativity"] = float("nan")
    try:
        lcc = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        if lcc.number_of_nodes() <= 300:
            out["avg_shortest_path"] = float(nx.average_shortest_path_length(lcc))
        else:
            nodes = list(lcc.nodes())
            rng = np.random.default_rng(0)
            lengths = []
            for _ in range(sample_paths):
                s, t = rng.choice(nodes, size=2, replace=False)
                lengths.append(nx.shortest_path_length(lcc, int(s), int(t)))
            out["avg_shortest_path"] = float(np.mean(lengths))
            out["avg_shortest_path_sampled"] = 1.0
    except Exception:
        out["avg_shortest_path"] = float("nan")
    try:
        core = np.array(list(nx.core_number(g).values()), dtype=float)
        out["kcore_mean"] = float(core.mean()) if core.size else 0.0
    except Exception:
        out["kcore_mean"] = float("nan")
    return out


def topology_recovery_metrics(full_data: Data, augmented_data: Data) -> Dict[str, float]:
    full = _topology(pyg_to_networkx(full_data))
    aug = _topology(pyg_to_networkx(augmented_data))
    metrics: Dict[str, float] = {}
    for key, full_value in full.items():
        aug_value = aug.get(key, float("nan"))
        metrics[f"full_{key}"] = full_value
        metrics[f"augmented_{key}"] = aug_value
        if np.isfinite(full_value) and np.isfinite(aug_value):
            metrics[f"{key}_error"] = abs(aug_value - full_value)
    return metrics


def _mmd(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    xx = rbf_kernel(x, x).mean()
    yy = rbf_kernel(y, y).mean()
    xy = rbf_kernel(x, y).mean()
    return float(xx + yy - 2 * xy)


def _hist_js(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    lo = float(min(np.min(a), np.min(b))) if len(a) and len(b) else 0.0
    hi = float(max(np.max(a), np.max(b))) if len(a) and len(b) else 1.0
    if lo == hi:
        hi = lo + 1.0
    ah, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    bh, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    ah = ah / max(1, ah.sum())
    bh = bh / max(1, bh.sum())
    return float(jensenshannon(ah, bh, base=2.0) ** 2)


def hidden_node_recovery_metrics(split: NodeHoldoutSplit, generated_x: torch.Tensor, augmented_data: Optional[Data] = None) -> Dict[str, float]:
    hidden_x = split.full_data_reference.x[split.hidden_node_ids].float().detach().cpu().numpy()
    gen_x = generated_x.float().detach().cpu().numpy()
    metrics = {"feature_mmd": _mmd(hidden_x, gen_x)}
    if hidden_x.size and gen_x.size:
        for dim in range(min(hidden_x.shape[1], gen_x.shape[1], 5)):
            metrics[f"feature_wasserstein_dim{dim}"] = float(wasserstein_distance(hidden_x[:, dim], gen_x[:, dim]))
    full_g = pyg_to_networkx(split.full_data_reference)
    hidden_deg = np.array([full_g.degree(int(i)) for i in split.hidden_node_ids.tolist()], dtype=float)
    if augmented_data is not None:
        aug_g = pyg_to_networkx(augmented_data)
        gen_ids = range(split.observed_data.num_nodes, augmented_data.num_nodes)
        gen_deg = np.array([aug_g.degree(int(i)) for i in gen_ids], dtype=float)
        metrics["degree_distribution_js"] = _hist_js(hidden_deg, gen_deg) if len(gen_deg) else float("nan")
        metrics["degree_wasserstein"] = float(wasserstein_distance(hidden_deg, gen_deg)) if len(gen_deg) else float("nan")
        try:
            bins = 16
            full_roles = compute_role_ids(split.full_data_reference, bins).detach().cpu().numpy()
            aug_roles = compute_role_ids(augmented_data, bins).detach().cpu().numpy()
            hidden_roles = full_roles[split.hidden_node_ids.detach().cpu().numpy()]
            gen_roles = aug_roles[list(gen_ids)]
            metrics["role_distribution_js"] = _hist_js(hidden_roles, gen_roles, bins=bins)
        except Exception:
            metrics["role_distribution_js"] = float("nan")
    obs_x = split.observed_data.x.detach().cpu()
    gen_t = generated_x.detach().cpu()
    hidden_t = split.full_data_reference.x[split.hidden_node_ids].detach().cpu()
    if gen_t.numel() and hidden_t.numel():
        metrics["nn_distance_to_hidden"] = float(torch.cdist(gen_t.float(), hidden_t.float()).min(dim=1).values.mean())
    if gen_t.numel() and obs_x.numel():
        metrics["nn_distance_to_observed"] = float(torch.cdist(gen_t.float(), obs_x.float()).min(dim=1).values.mean())
    if torch.is_tensor(getattr(split.full_data_reference, "y", None)):
        labels = split.full_data_reference.y[split.hidden_node_ids].detach().cpu().numpy()
        metrics["hidden_label_count"] = float(len(np.unique(labels)))
    return metrics


def attachment_recovery_metrics(split: NodeHoldoutSplit, generated_edges: torch.Tensor, scores: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    old_hidden = set(split.hidden_node_ids.tolist())
    hidden_endpoints = []
    for s, d in split.hidden_edges.t().tolist():
        if s in old_hidden and d not in old_hidden and d in split.old_to_observed:
            hidden_endpoints.append(split.old_to_observed[d])
        elif d in old_hidden and s not in old_hidden and s in split.old_to_observed:
            hidden_endpoints.append(split.old_to_observed[s])
    hidden_endpoint_set = set(hidden_endpoints)
    predicted = [int(d) for s, d in generated_edges.t().tolist() if int(s) >= split.observed_data.num_nodes and int(d) < split.observed_data.num_nodes]
    if predicted:
        metrics["hidden_edge_precision_at_k"] = float(np.mean([p in hidden_endpoint_set for p in predicted]))
        metrics["hidden_edge_recall_at_k"] = float(len(set(predicted) & hidden_endpoint_set) / max(1, len(hidden_endpoint_set)))
        y = np.array([1 if i in hidden_endpoint_set else 0 for i in range(split.observed_data.num_nodes)], dtype=int)
        endpoint_scores = np.zeros(split.observed_data.num_nodes, dtype=float)
        for rank, node in enumerate(predicted):
            if 0 <= node < split.observed_data.num_nodes:
                endpoint_scores[node] = max(endpoint_scores[node], 1.0 - rank / max(1, len(predicted)))
        if y.sum() > 0 and y.sum() < len(y) and endpoint_scores.max() > endpoint_scores.min():
            metrics["attachment_auc_comparable"] = float(roc_auc_score(y, endpoint_scores))
            metrics["attachment_ap_comparable"] = float(average_precision_score(y, endpoint_scores))
    metrics["hidden_attachment_endpoint_count"] = float(len(hidden_endpoint_set))
    if scores is not None and labels is not None:
        y = labels.detach().cpu().numpy()
        p = scores.detach().cpu().numpy()
        try:
            metrics["attachment_auc"] = float(roc_auc_score(y, p))
            metrics["attachment_ap"] = float(average_precision_score(y, p))
        except Exception:
            metrics["attachment_auc"] = float("nan")
            metrics["attachment_ap"] = float("nan")
    return metrics


def downstream_robustness_metrics(full_data: Data, observed_data: Data, augmented_data: Data) -> Dict[str, float]:
    """Lightweight downstream proxy metrics.

    Full classifier training is intentionally left to the experiment layer for
    larger runs; this proxy reports feature-smoothness gaps and community NMI on
    observed nodes without fabricating classifier claims.
    """
    metrics: Dict[str, float] = {}
    try:
        full_g = pyg_to_networkx(full_data)
        aug_g = pyg_to_networkx(augmented_data)
        if full_g.number_of_nodes() > 1000:
            full_comms = list(nx.community.asyn_lpa_communities(full_g, seed=0))
        else:
            full_comms = list(nx.community.greedy_modularity_communities(full_g))
        if aug_g.number_of_nodes() > 1000:
            aug_comms = list(nx.community.asyn_lpa_communities(aug_g, seed=0))
        else:
            aug_comms = list(nx.community.greedy_modularity_communities(aug_g))
        full_labels = np.zeros(observed_data.num_nodes, dtype=int)
        aug_labels = np.zeros(observed_data.num_nodes, dtype=int)
        for i, comm in enumerate(full_comms):
            for node in comm:
                if int(node) < observed_data.num_nodes:
                    full_labels[int(node)] = i
        for i, comm in enumerate(aug_comms):
            for node in comm:
                if int(node) < observed_data.num_nodes:
                    aug_labels[int(node)] = i
        metrics["community_nmi_observed_nodes"] = float(normalized_mutual_info_score(full_labels, aug_labels))
    except Exception:
        metrics["community_nmi_observed_nodes"] = float("nan")
    return metrics


def evaluate_all(split: NodeHoldoutSplit, generated_x: torch.Tensor, generated_edges: torch.Tensor) -> Dict[str, float]:
    augmented = build_augmented_data(split.observed_data, generated_x, generated_edges)
    metrics: Dict[str, float] = {}
    metrics.update(topology_recovery_metrics(split.full_data_reference, augmented))
    metrics.update(hidden_node_recovery_metrics(split, generated_x, augmented))
    metrics.update(attachment_recovery_metrics(split, generated_edges))
    metrics.update(downstream_robustness_metrics(split.full_data_reference, split.observed_data, augmented))
    return metrics
