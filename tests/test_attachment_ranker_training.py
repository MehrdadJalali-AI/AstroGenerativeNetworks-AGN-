import torch
from torch_geometric.data import Data

from agn_real.models.attachment_ranker import attach_generated_nodes, train_attachment_ranker


def test_attachment_ranker_training_and_inference():
    data = Data(
        x=torch.rand(10, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long),
        num_nodes=10,
    )
    ranker, info = train_attachment_ranker(data, config={"epochs": 2, "seed": 0})
    edges = attach_generated_nodes(data, torch.rand(2, 4), ranker=ranker, mode="learned_ranker", k=2, threshold=0.0)
    assert info["pair_dim"] == 6
    assert edges.shape[0] == 2
    assert edges.shape[1] > 0

