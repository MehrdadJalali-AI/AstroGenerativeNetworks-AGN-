from pathlib import Path

import pandas as pd


def test_final_manuscript_tables_have_one_agn_label_if_present():
    path = Path("results/final/tables/MANUSCRIPT_Table_2_main_node_holdout.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "model" not in df.columns:
        return
    assert "rc_agn" not in set(df["model"])
    assert "standard_agn" not in set(df["model"])
    assert "Unconditioned + cosine" in set(df["model"])
    assert "AGN" in set(df["model"])

