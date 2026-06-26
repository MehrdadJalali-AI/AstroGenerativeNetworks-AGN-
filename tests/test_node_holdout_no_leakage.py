import torch
from torch_geometric.data import Data

from agn_real.splits.node_holdout import stratified_node_holdout


def test_hidden_nodes_and_incident_edges_absent_from_observed_graph():
    data = Data(
        x=torch.randn(6, 3),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 0, 2, 4, 1], [1, 2, 3, 4, 5, 0, 2, 4, 1, 3]],
            dtype=torch.long,
        ),
        num_nodes=6,
    )
    split = stratified_node_holdout(data, 0.33, seed=1, strategy="random")
    hidden = set(split.hidden_node_ids.tolist())
    assert split.observed_data.num_nodes == data.num_nodes - len(hidden)
    for old_src, old_dst in data.edge_index.t().tolist():
        if old_src in hidden or old_dst in hidden:
            continue
        assert old_src in split.old_to_observed
        assert old_dst in split.old_to_observed
    for new_src, new_dst in split.observed_data.edge_index.t().tolist():
        assert split.observed_to_old[int(new_src)] not in hidden
        assert split.observed_to_old[int(new_dst)] not in hidden

