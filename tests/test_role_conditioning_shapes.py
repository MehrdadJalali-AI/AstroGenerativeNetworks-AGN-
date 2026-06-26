import torch
from torch_geometric.data import Data

from agn_real.models.rc_agn import RCAGNConfig, compute_role_ids, generate_nodes, train_model


def test_role_conditioning_shapes():
    data = Data(
        x=torch.rand(8, 5),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long),
        num_nodes=8,
    )
    cfg = RCAGNConfig(input_dim=5, hidden_dim=8, latent_dim=4, num_role_bins=6, epochs=2, seed=0)
    model, info = train_model(data, cfg)
    roles = compute_role_ids(data, cfg.num_role_bins)
    gen = generate_nodes(model, 3, torch.bincount(roles, minlength=cfg.num_role_bins).float(), cfg)
    assert gen["x"].shape == (3, 5)
    assert gen["z"].shape == (3, 4)
    assert gen["roles"].shape == (3,)

