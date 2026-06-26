from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from agn_real.data import load_real_dataset
from agn_real.eval.metrics import evaluate_all
from agn_real.experiments.run_real_node_holdout import run_rc_agn_ablation
from agn_real.splits import stratified_node_holdout
from agn_real.utils import apply_feature_mode, ensure_dir, set_seed


ABLATION_MODELS = [
    "agn_current",
    "agn_preferential_blend",
    "agn_feature_mmd",
    "agn_feature_centroid",
    "agn_role_interpolation",
    "agn_mmd_preferential",
    "agn_mmd_centroid_preferential",
    "agn_hybrid_feature_hybrid_attachment_validation",
]

DISPLAY = {
    "agn_current": "AGN_current",
    "agn_preferential_blend": "AGN + preferential blend",
    "agn_feature_mmd": "AGN + feature MMD loss",
    "agn_feature_centroid": "AGN + feature centroid regularization",
    "agn_role_interpolation": "AGN + role interpolation feature generator",
    "agn_mmd_preferential": "AGN + feature MMD + preferential blend",
    "agn_mmd_centroid_preferential": "AGN + feature MMD + centroid + preferential blend",
    "agn_hybrid_feature_hybrid_attachment_validation": "AGN + hybrid feature + hybrid attachment + validation selection",
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AGN improvement ablation grid.")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default="results/final")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--holdout_ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--models", nargs="+", default=ABLATION_MODELS)
    parser.add_argument("--feature_mode", default="raw")
    parser.add_argument("--split_strategy", default="degree_stratified")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--attachment_epochs", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--encoder", default="gcn", choices=["gcn", "sage", "gat"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--attachment_threshold", type=float, default=0.0)
    parser.add_argument("--agn_feature_mode", default="hybrid_feature")
    parser.add_argument("--agn_attachment_mode", default="hybrid")
    parser.add_argument("--feature_mmd_weight", type=float, default=0.5)
    parser.add_argument("--feature_centroid_weight", type=float, default=0.5)
    parser.add_argument("--interpolation_lambda", type=float, default=0.5)
    parser.add_argument("--hybrid_alpha", type=float, default=0.6)
    parser.add_argument("--hybrid_beta", type=float, default=0.2)
    parser.add_argument("--hybrid_gamma", type=float, default=0.1)
    parser.add_argument("--hybrid_delta", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.dataset_root:
        args.data_root = args.dataset_root

    out = Path(args.output_dir)
    raw_dir = ensure_dir(out / "raw")
    summary_dir = ensure_dir(out / "summary")
    table_dir = ensure_dir(out / "tables")
    rows = []
    for dataset in args.datasets:
        base, _ = load_real_dataset(dataset, args.data_root)
        data = apply_feature_mode(base, args.feature_mode)
        for ratio in args.holdout_ratios:
            for seed in args.seeds:
                set_seed(seed)
                split = stratified_node_holdout(data, ratio, seed, args.split_strategy)
                for model_name in args.models:
                    row = {"dataset": dataset, "holdout_ratio": ratio, "seed": seed, "model": model_name, "model_label": DISPLAY.get(model_name, model_name)}
                    try:
                        result = run_rc_agn_ablation(model_name, split.observed_data, len(split.hidden_node_ids), seed, args)
                        row["status"] = result["status"]
                        if result["status"] == "ok":
                            row.update(evaluate_all(split, result["generated_x"], result["edge_index"]))
                            row["agn_feature_mode"] = result.get("agn_feature_mode")
                            row["agn_attachment_mode"] = result.get("agn_attachment_mode")
                        else:
                            row["reason"] = result.get("reason", "unavailable")
                    except Exception as exc:
                        row["status"] = "not_run"
                        row["reason"] = str(exc)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(raw_dir / "agn_improvement_ablation_raw.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_dir / "agn_improvement_ablation_raw.csv", index=False)
    metrics = [
        "feature_mmd",
        "hidden_edge_precision_at_k",
        "attachment_auc_comparable",
        "attachment_ap_comparable",
        "density_error",
        "average_degree_error",
    ]
    ok = raw[raw["status"] == "ok"]
    summary = ok.groupby(["dataset", "holdout_ratio", "model", "model_label"], dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c != ""]).rstrip("_") for col in summary.columns]
    summary.to_csv(summary_dir / "agn_improvement_ablation.csv", index=False)
    summary.to_csv(table_dir / "MANUSCRIPT_Table_AGN_ablation.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
