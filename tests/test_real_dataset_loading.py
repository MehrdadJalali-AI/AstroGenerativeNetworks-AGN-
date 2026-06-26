import torch
from torch_geometric.data import Data

from agn_real.data.loaders import summarize_dataset


def test_summarize_dataset_uses_raw_features():
    data = Data(
        x=torch.eye(4),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        num_nodes=4,
    )
    summary = summarize_dataset(data)
    assert summary["num_nodes"] == 4
    assert summary["feature_dim"] == 4
    assert summary["num_classes"] == 2

