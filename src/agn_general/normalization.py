"""
Two-stage node-feature normalization: z-score (StandardScaler) then per-dimension
min--max to [0, 1], with inverse mapping back to raw feature space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureNormalizer:
    """Stores parameters for normalize / denormalize."""

    scaler: StandardScaler
    feat_min: np.ndarray
    feat_max: np.ndarray

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """Map [0,1]^d normalized features back to raw space (inverse min--max on z-scored coords, then inverse z-score)."""
        x_norm = np.asarray(x_norm, dtype=np.float64)
        rng = self.feat_max - self.feat_min
        rng = np.where(rng == 0, 1.0, rng)
        x_std = x_norm * rng + self.feat_min
        return self.scaler.inverse_transform(x_std).astype(np.float32)


def fit_two_stage_normalize(features: np.ndarray) -> Tuple[np.ndarray, FeatureNormalizer]:
    """
    Fit z-score then per-feature min--max to [0, 1] using column-wise min/max on standardized values.

    Returns:
        features_scaled: array in [0, 1]^d (float32)
        normalizer: for denormalize()
    """
    features = np.asarray(features, dtype=np.float64)
    scaler = StandardScaler()
    x_std = scaler.fit_transform(features)
    feat_min = x_std.min(axis=0)
    feat_max = x_std.max(axis=0)
    rng = feat_max - feat_min
    rng = np.where(rng == 0, 1.0, rng)
    x_norm = (x_std - feat_min) / rng
    normalizer = FeatureNormalizer(
        scaler=scaler,
        feat_min=feat_min.astype(np.float64),
        feat_max=feat_max.astype(np.float64),
    )
    return x_norm.astype(np.float32), normalizer


def tuple_from_normalizer(n: FeatureNormalizer):
    """Backward-compatible (scaler, feat_min, feat_max) for legacy call sites."""
    return n.scaler, n.feat_min.astype(np.float32), n.feat_max.astype(np.float32)


def denormalize_from_parts(
    x_norm: np.ndarray,
    scaler: StandardScaler,
    feat_min: np.ndarray,
    feat_max: np.ndarray,
) -> np.ndarray:
    """Inverse min--max on standardized coordinates, then inverse z-score (raw scale)."""
    n = FeatureNormalizer(
        scaler=scaler,
        feat_min=np.asarray(feat_min, dtype=np.float64),
        feat_max=np.asarray(feat_max, dtype=np.float64),
    )
    return n.denormalize(x_norm)
