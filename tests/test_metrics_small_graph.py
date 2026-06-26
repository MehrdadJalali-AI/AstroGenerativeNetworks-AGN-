import torch
from torch_geometric.data import Data

from agn_real.eval.metrics import build_augmented_data, topology_recovery_metrics


def test_metrics_small_graph():
    full = Data(x=torch.rand(4, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long), num_nodes=4)
    observed = Data(x=torch.rand(3, 3), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long), num_nodes=3)
    aug = build_augmented_data(observed, torch.rand(1, 3), torch.tensor([[3, 1], [1, 3]], dtype=torch.long))
    metrics = topology_recovery_metrics(full, aug)
    assert "density_error" in metrics
    assert aug.num_nodes == 4

