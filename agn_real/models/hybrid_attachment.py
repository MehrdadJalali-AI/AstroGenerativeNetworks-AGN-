from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from agn_real.models.attachment_ranker import AttachmentRanker, build_pair_features, node_roles
from agn_real.utils import pyg_to_networkx


@dataclass
class HybridAttachmentConfig:
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1
    delta: float = 0.0
    normalize_weights: bool = True
    k: int = 10
    threshold: float = 0.0

    def normalized(self) -> "HybridAttachmentConfig":
        if not self.normalize_weights:
            return self
        total = float(self.alpha + self.beta + self.gamma + self.delta)
        if total <= 0:
            return HybridAttachmentConfig(k=self.k, threshold=self.threshold)
        return HybridAttachmentConfig(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
            delta=self.delta / total,
            normalize_weights=False,
            k=self.k,
            threshold=self.threshold,
        )


def _preferential_scores(observed_data: Data) -> torch.Tensor:
    g = pyg_to_networkx(observed_data)
    deg = torch.tensor([g.degree(i) + 1.0 for i in range(observed_data.num_nodes)], dtype=torch.float32)
    return deg / deg.max().clamp_min(1.0)


def _role_similarity_scores(roles: Dict[str, torch.Tensor], generated_role: Optional[int], n_obs: int) -> torch.Tensor:
    if generated_role is None:
        comm = roles["community"][:n_obs].float()
        if comm.numel() == 0:
            return torch.zeros(n_obs)
        counts = torch.bincount(comm.long().clamp_min(0))
        major = int(torch.argmax(counts).item()) if counts.numel() else 0
    else:
        major = int(generated_role)
    comm_match = (roles["community"][:n_obs].round().long() == major).float()
    core = roles["core"][:n_obs].float()
    core_score = core / core.max().clamp_min(1.0)
    return 0.5 * comm_match + 0.5 * core_score


def hybrid_attachment_scores(
    observed_data: Data,
    generated_x: torch.Tensor,
    generated_z: Optional[torch.Tensor],
    ranker: Optional[AttachmentRanker],
    config: HybridAttachmentConfig,
    generated_roles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cfg = config.normalized()
    n_obs = observed_data.num_nodes
    n_gen = generated_x.size(0)
    obs_x = observed_data.x.detach().float().cpu()
    all_x = torch.cat([obs_x, generated_x.detach().float().cpu()], dim=0)
    all_z = None
    if generated_z is not None:
        all_z = torch.cat([torch.zeros((n_obs, generated_z.size(1))), generated_z.detach().float().cpu()], dim=0)
    roles = node_roles(observed_data)
    for key in roles:
        roles[key] = roles[key].detach().cpu()
        roles[key] = torch.cat([roles[key], torch.zeros(n_gen, dtype=roles[key].dtype)])
    pref = _preferential_scores(observed_data)
    out = torch.zeros((n_gen, n_obs), dtype=torch.float32)
    ranker_device = next(ranker.parameters()).device if ranker is not None else torch.device("cpu")
    if ranker is not None:
        ranker.eval()
    for gi in range(n_gen):
        src = torch.full((n_obs,), n_obs + gi, dtype=torch.long)
        dst = torch.arange(n_obs, dtype=torch.long)
        if ranker is not None and cfg.alpha > 0:
            pair = build_pair_features(all_x, all_z, roles, src, dst).to(ranker_device)
            with torch.no_grad():
                learned = ranker.predict_proba(pair).detach().cpu()
        else:
            learned = torch.zeros(n_obs)
        gen_role = int(generated_roles[gi].item()) if generated_roles is not None and gi < len(generated_roles) else None
        role_score = _role_similarity_scores(roles, gen_role, n_obs)
        feature_score = F.cosine_similarity(generated_x[gi].float().view(1, -1), obs_x.float(), dim=1)
        feature_score = (feature_score + 1.0) / 2.0
        out[gi] = cfg.alpha * learned + cfg.beta * pref + cfg.gamma * role_score + cfg.delta * feature_score
    return out.clamp(0.0, 1.0)


def attach_hybrid_generated_nodes(
    observed_data: Data,
    generated_x: torch.Tensor,
    generated_z: Optional[torch.Tensor],
    ranker: Optional[AttachmentRanker],
    config: HybridAttachmentConfig,
    generated_roles: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = hybrid_attachment_scores(observed_data, generated_x, generated_z, ranker, config, generated_roles)
    n_obs = observed_data.num_nodes
    edges = []
    edge_scores = []
    for gi in range(scores.size(0)):
        top = torch.topk(scores[gi], k=min(config.k, n_obs)).indices.tolist()
        for target in top:
            score = float(scores[gi, target])
            if score >= config.threshold or len(top) <= config.k:
                edges.append((n_obs + gi, int(target)))
                edge_scores.append(score)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    score_t = torch.tensor(edge_scores, dtype=torch.float32)
    return undirected, torch.cat([score_t, score_t], dim=0)
