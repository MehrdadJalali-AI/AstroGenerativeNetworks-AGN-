from pathlib import Path

from agn_real.experiments.run_real_node_holdout import main


def test_cli_smoke_records_unavailable_dataset(tmp_path: Path):
    out = tmp_path / "results"
    code = main([
        "--datasets",
        "DefinitelyMissingDataset",
        "--seeds",
        "0",
        "--holdout_ratios",
        "0.1",
        "--feature_modes",
        "raw",
        "--models",
        "rc_agn",
        "--output_dir",
        str(out),
    ])
    assert code == 0
    assert (out / "raw" / "real_node_holdout_results.csv").exists()

