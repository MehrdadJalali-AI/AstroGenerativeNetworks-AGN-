import torch
from torch_geometric.data import Data

from agn_real.splits import stratified_node_holdout


def test_ppi_like_holdout_no_leakage():
    data = Data(
        x=torch.rand(20, 6),
        y=torch.randint(0, 2, (20, 4)).float(),
        edge_index=torch.tensor([[i for i in range(20)], [(i + 1) % 20 for i in range(20)]], dtype=torch.long),
        num_nodes=20,
    )
    split = stratified_node_holdout(data, 0.2, 0, "random")
    hidden = set(split.hidden_node_ids.tolist())
    for s, d in split.observed_data.edge_index.t().tolist():
        assert split.observed_to_old[int(s)] not in hidden
        assert split.observed_to_old[int(d)] not in hidden

