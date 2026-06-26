import argparse

import torch
from torch_geometric.data import Data

from agn_real.experiments.validation_selection import select_on_observed_validation
from agn_real.splits import stratified_node_holdout


def test_validation_selection_receives_only_observed_graph_nodes():
    data = Data(
        x=torch.rand(30, 4),
        y=torch.arange(30) % 3,
        edge_index=torch.tensor([list(range(29)) + list(range(1, 30)), list(range(1, 30)) + list(range(29))], dtype=torch.long),
        num_nodes=30,
    )
    split = stratified_node_holdout(data, 0.2, 0, "degree_stratified")
    args = argparse.Namespace(
        validation_holdout_ratio=0.1,
        epochs=1,
        validation_epochs=1,
        attachment_epochs=1,
        validation_attachment_epochs=1,
        hidden_dim=8,
        latent_dim=4,
        encoder="gcn",
        k=2,
        attachment_threshold=0.0,
        device="cpu",
        validation_precision_weight=1.0,
        validation_ap_weight=1.0,
        validation_mmd_weight=1.0,
    )
    _, _, rows = select_on_observed_validation(split.observed_data, 0, args)
    assert split.observed_data.num_nodes < data.num_nodes
    assert rows
