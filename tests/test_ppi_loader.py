import torch
from torch_geometric.data import Data

import agn_real.data.load_ppi as ppi_mod


def test_ppi_loader_supports_selected_graph_and_max_nodes(monkeypatch, tmp_path):
    class FakePPI:
        def __init__(self, root, split="train"):
            self.data = [Data(x=torch.rand(30, 5), y=torch.rand(30, 3), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), num_nodes=30)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    monkeypatch.setattr(ppi_mod, "PPI", FakePPI)
    data, meta = ppi_mod.load_ppi_dataset(tmp_path / "ppi", ppi_root=tmp_path / "ppi", max_nodes=20, allow_download=True)
    assert data.num_nodes == 20
    assert data.x.size(0) == data.num_nodes
    assert meta["dataset"] == "PPI"
