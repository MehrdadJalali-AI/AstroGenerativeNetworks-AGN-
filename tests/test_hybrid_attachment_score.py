import torch
from torch_geometric.data import Data

from agn_real.models.hybrid_attachment import HybridAttachmentConfig, attach_hybrid_generated_nodes, hybrid_attachment_scores


def test_hybrid_attachment_preferential_component_favors_high_degree_nodes():
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]], dtype=torch.long)
    data = Data(x=torch.eye(4), edge_index=edge_index, num_nodes=4)
    generated_x = torch.ones(1, 4)
    cfg = HybridAttachmentConfig(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0, k=1, threshold=0.0)
    scores = hybrid_attachment_scores(data, generated_x, None, None, cfg)
    assert int(torch.argmax(scores[0]).item()) == 0
    edges, edge_scores = attach_hybrid_generated_nodes(data, generated_x, None, None, cfg)
    assert edges.shape[0] == 2
    assert edge_scores.numel() == edges.shape[1]
