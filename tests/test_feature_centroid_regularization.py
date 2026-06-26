import torch
from torch_geometric.data import Data

from agn_real.models.feature_generator import centroid_regularization_loss, compute_role_feature_stats


def test_centroid_regularization_is_lower_at_role_centroids():
    data = Data(x=torch.tensor([[1.0, 1.0], [1.0, 3.0], [5.0, 5.0], [7.0, 5.0]]), num_nodes=4)
    roles = torch.tensor([0, 0, 1, 1])
    stats = compute_role_feature_stats(data, roles, 2)
    at_centroids = stats.centroids[roles]
    far = at_centroids + 10.0
    assert centroid_regularization_loss(at_centroids, roles, stats) < centroid_regularization_loss(far, roles, stats)
