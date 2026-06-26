import torch
from torch_geometric.data import Data

from agn_real.eval.metrics import attachment_recovery_metrics
from agn_real.splits import stratified_node_holdout


def test_comparable_attachment_auc_for_predicted_edges():
    data = Data(
        x=torch.rand(12, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long),
        num_nodes=12,
    )
    split = stratified_node_holdout(data, 0.25, 1, "random")
    gen_start = split.observed_data.num_nodes
    endpoints = list(range(min(4, split.observed_data.num_nodes)))
    edges = torch.tensor([[gen_start] * len(endpoints), endpoints], dtype=torch.long)
    metrics = attachment_recovery_metrics(split, torch.cat([edges, edges.flip(0)], dim=1))
    assert "hidden_edge_precision_at_k" in metrics
    assert "hidden_edge_recall_at_k" in metrics

