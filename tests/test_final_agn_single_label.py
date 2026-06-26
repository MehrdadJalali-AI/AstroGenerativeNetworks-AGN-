from pathlib import Path

import pandas as pd


def test_main_tables_have_only_one_main_agn_label():
    path = Path("results/final/tables/MANUSCRIPT_Table_2_main_node_holdout.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    labels = set(df.get("model", []))
    assert "AGN" in labels
    assert "rc_agn" not in labels
    assert "standard_agn" not in labels
    assert "AGN_current" not in labels
