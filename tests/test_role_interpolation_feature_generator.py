import torch
from torch_geometric.data import Data

from agn_real.models.feature_generator import role_interpolated_features


def test_role_interpolation_uses_observed_role_similar_features_when_lambda_zero():
    data = Data(x=torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]]), num_nodes=3)
    observed_roles = torch.tensor([0, 0, 1])
    decoded = torch.zeros(2, 2)
    generated_roles = torch.tensor([0, 1])
    out = role_interpolated_features(decoded, generated_roles, data, observed_roles, interpolation_lambda=0.0, seed=0)
    assert out.shape == decoded.shape
    assert out[0, 1].item() == 0.0
    assert out[1, 1].item() == 5.0
