import torch

from agn_real.eval.downstream import dropout_generated_edges, prune_generated_edges


def test_generated_edge_pruning_keeps_high_confidence_edges_only():
    edge_index = torch.tensor([[5, 5, 1], [0, 2, 2]], dtype=torch.long)
    scores = torch.tensor([0.9, 0.1, 1.0])
    pruned = prune_generated_edges(edge_index, scores, observed_num_nodes=5, threshold=0.5)
    assert pruned.shape[1] == 2
    assert [5, 0] in pruned.t().tolist()
    assert [1, 2] in pruned.t().tolist()


def test_generated_edge_dropout_preserves_observed_edges():
    edge_index = torch.tensor([[5, 5, 1], [0, 2, 2]], dtype=torch.long)
    dropped = dropout_generated_edges(edge_index, observed_num_nodes=5, p=1.0, seed=0)
    assert dropped.t().tolist() == [[1, 2]]
