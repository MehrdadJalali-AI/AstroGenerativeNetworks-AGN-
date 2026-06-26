from pathlib import Path

import pandas as pd


def test_sensitivity_includes_meaningful_metrics_if_present():
    path = Path("results/final/summary/sensitivity_curves.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert "metric" in df.columns
    assert {"hidden_edge_precision_at_k", "feature_mmd", "density_error"}.intersection(set(df["metric"]))

