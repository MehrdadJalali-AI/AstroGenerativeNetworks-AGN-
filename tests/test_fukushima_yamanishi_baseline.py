import torch
from torch_geometric.data import Data

from agn_real.baselines.fukushima_yamanishi_gca import run_fukushima_yamanishi_gca


def test_fukushima_yamanishi_style_baseline_runs_on_small_graph():
    data = Data(
        x=torch.rand(12, 5),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6]],
            dtype=torch.long,
        ),
        num_nodes=12,
    )
    out = run_fukushima_yamanishi_gca(data, num_nodes=3, seed=0, latent_dim=3, k=2, threshold=0.0)
    assert out["status"] == "ok"
    assert out["generated_x"].shape == (3, 5)
    assert out["edge_index"].shape[0] == 2

