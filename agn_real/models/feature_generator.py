from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from agn_real.models.rc_agn import compute_role_ids


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Multi-kernel RBF maximum mean discrepancy."""
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)
    x = x.float()
    y = y.float()
    if sigmas is None:
        sigmas = x.new_tensor([0.5, 1.0, 2.0, 4.0, 8.0])
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    loss = x.new_tensor(0.0)
    for sigma in sigmas.to(x.device).float():
        gamma = 1.0 / (2.0 * sigma.pow(2).clamp_min(1e-12))
        loss = loss + torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2.0 * torch.exp(-gamma * xy).mean()
    return loss / sigmas.numel()


@dataclass
class RoleFeatureStats:
    centroids: torch.Tensor
    spreads: torch.Tensor
    counts: torch.Tensor


def compute_role_feature_stats(data: Data, roles: torch.Tensor, num_roles: int) -> RoleFeatureStats:
    x = data.x.detach().float().cpu()
    roles = roles.detach().cpu().long()
    centroids = torch.zeros((num_roles, x.size(1)), dtype=torch.float32)
    spreads = torch.ones((num_roles, x.size(1)), dtype=torch.float32)
    counts = torch.zeros(num_roles, dtype=torch.long)
    global_centroid = x.mean(dim=0)
    global_spread = x.std(dim=0).clamp_min(1e-6)
    for role in range(num_roles):
        mask = roles == role
        counts[role] = int(mask.sum())
        if mask.any():
            vals = x[mask]
            centroids[role] = vals.mean(dim=0)
            spreads[role] = vals.std(dim=0).clamp_min(1e-6) if vals.size(0) > 1 else global_spread
        else:
            centroids[role] = global_centroid
            spreads[role] = global_spread
    return RoleFeatureStats(centroids=centroids, spreads=spreads, counts=counts)


def centroid_regularization_loss(generated_x: torch.Tensor, generated_roles: torch.Tensor, stats: RoleFeatureStats) -> torch.Tensor:
    centroids = stats.centroids.to(generated_x.device)[generated_roles.long().to(generated_x.device)]
    return F.mse_loss(generated_x.float(), centroids.float())


def role_interpolated_features(
    decoded_x: torch.Tensor,
    generated_roles: torch.Tensor,
    observed_data: Data,
    observed_roles: torch.Tensor,
    interpolation_lambda: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Blend decoded features with prototypes sampled from observed nodes in the same role."""
    if decoded_x.numel() == 0:
        return decoded_x
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    obs_x = observed_data.x.detach().float().cpu()
    obs_roles = observed_roles.detach().cpu().long()
    gen_roles = generated_roles.detach().cpu().long()
    prototypes = []
    for role in gen_roles.tolist():
        candidates = torch.where(obs_roles == int(role))[0]
        if candidates.numel() == 0:
            candidates = torch.arange(obs_x.size(0))
        idx = candidates[torch.randint(candidates.numel(), (1,), generator=rng)].item()
        prototypes.append(obs_x[idx])
    proto = torch.stack(prototypes, dim=0).to(decoded_x.device)
    lam = float(max(0.0, min(1.0, interpolation_lambda)))
    return lam * decoded_x.float() + (1.0 - lam) * proto.float()


def role_distribution_from_data(data: Data, num_role_bins: int) -> torch.Tensor:
    roles = compute_role_ids(data, num_role_bins)
    return torch.bincount(roles.cpu(), minlength=num_role_bins).float()


def feature_mode_uses_interpolation(feature_mode: str) -> bool:
    return feature_mode in {"role_interpolation", "hybrid_feature"}


def feature_mode_uses_mmd(feature_mode: str) -> bool:
    return feature_mode in {"decoder_mmd", "hybrid_feature"}


def feature_mode_uses_centroid(feature_mode: str) -> bool:
    return feature_mode in {"centroid_regularized", "hybrid_feature"}
