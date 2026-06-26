from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agn_real.baselines import run_baseline
from agn_real.data import load_real_dataset
from agn_real.eval.downstream import ClassifierConfig, dropout_generated_edges, prune_generated_edges, random_label_split, train_and_evaluate_node_classifier
from agn_real.eval.metrics import build_augmented_data
from agn_real.experiments.run_real_node_holdout import run_rc_agn
from agn_real.splits import stratified_node_holdout
from agn_real.utils import apply_feature_mode, ensure_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run downstream node classification robustness on real networks.")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed", "PPI"])
    parser.add_argument("--models", nargs="+", default=["rc_agn", "fukushima_yamanishi_gca", "standard_agn", "knn_raw", "preferential", "random"])
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--ppi_root", default=None)
    parser.add_argument("--ppi_mode", default="selected_graph", choices=["selected_graph", "disjoint_union"])
    parser.add_argument("--ppi_graph_index", type=int, default=0)
    parser.add_argument("--ppi_max_nodes", type=int, default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--holdout_ratio", type=float, default=0.10)
    parser.add_argument("--feature_mode", default="raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_epochs", type=int, default=30)
    parser.add_argument("--attachment_epochs", type=int, default=15)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--encoder", default="gcn", choices=["gcn", "sage", "gat"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--attachment_threshold", type=float, default=0.5)
    parser.add_argument("--agn_feature_mode", default="hybrid_feature", choices=["decoder", "decoder_mmd", "centroid_regularized", "role_interpolation", "hybrid_feature"])
    parser.add_argument("--agn_attachment_mode", default="hybrid", choices=["learned_ranker", "hybrid"])
    parser.add_argument("--feature_mmd_weight", type=float, default=0.5)
    parser.add_argument("--feature_centroid_weight", type=float, default=0.5)
    parser.add_argument("--interpolation_lambda", type=float, default=0.5)
    parser.add_argument("--hybrid_alpha", type=float, default=0.6)
    parser.add_argument("--hybrid_beta", type=float, default=0.2)
    parser.add_argument("--hybrid_gamma", type=float, default=0.1)
    parser.add_argument("--hybrid_delta", type=float, default=0.1)
    parser.add_argument("--generated_edge_pruning_threshold", type=float, default=None)
    parser.add_argument("--generated_edge_top_k", type=int, default=None)
    parser.add_argument("--generated_edge_dropout", type=float, default=0.0)
    parser.add_argument("--use_edge_weight_if_supported", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.dataset_root:
        args.data_root = args.dataset_root
    rows = []
    raw_dir = ensure_dir(Path(args.output_dir) / "raw")
    summary_dir = ensure_dir(Path(args.output_dir) / "summary")
    for dataset in args.datasets:
        try:
            base, _ = load_real_dataset(
                dataset,
                args.data_root,
                ppi_root=args.ppi_root,
                ppi_mode=args.ppi_mode,
                ppi_graph_index=args.ppi_graph_index,
                ppi_max_nodes=args.ppi_max_nodes,
            )
            data = apply_feature_mode(base, args.feature_mode)
        except Exception as exc:
            rows.append({"dataset": dataset, "model": "all", "status": "not_run", "reason": f"dataset load failed: {exc}"})
            continue
        if not torch.is_tensor(getattr(data, "y", None)):
            rows.append({"dataset": dataset, "model": "all", "status": "not_run", "reason": "no labels"})
            continue
        for seed in args.seeds:
            split = stratified_node_holdout(data, args.holdout_ratio, seed, "degree_stratified")
            train_mask, _, test_mask = random_label_split(split.observed_data.num_nodes, seed)
            cfg = ClassifierConfig(epochs=args.epochs, seed=seed, device=args.device)
            observed_metrics = train_and_evaluate_node_classifier(split.observed_data, train_mask, test_mask, cfg)
            full_train, _, full_test = random_label_split(data.num_nodes, seed)
            full_metrics = train_and_evaluate_node_classifier(data, full_train, full_test, cfg)
            for model_name in ["observed_incomplete", *args.models]:
                row = {"dataset": dataset, "seed": seed, "model": model_name, "holdout_ratio": args.holdout_ratio, "feature_mode": args.feature_mode}
                if model_name == "observed_incomplete":
                    aug_metrics = observed_metrics
                else:
                    try:
                        if model_name in {"rc_agn", "agn"}:
                            result = run_rc_agn(split.observed_data, len(split.hidden_node_ids), seed, args)
                        else:
                            result = run_baseline(model_name, split.observed_data, len(split.hidden_node_ids), seed=seed, k=args.k, threshold=args.attachment_threshold, epochs=args.model_epochs)
                        if result["status"] != "ok":
                            row.update({"status": result["status"], "reason": result.get("reason", "unavailable")})
                            rows.append(row)
                            continue
                        edge_index = prune_generated_edges(
                            result["edge_index"],
                            result.get("edge_scores"),
                            split.observed_data.num_nodes,
                            threshold=args.generated_edge_pruning_threshold,
                            top_k=args.generated_edge_top_k,
                        )
                        edge_index = dropout_generated_edges(edge_index, split.observed_data.num_nodes, args.generated_edge_dropout, seed)
                        augmented = build_augmented_data(split.observed_data, result["generated_x"], edge_index)
                        gen_pad = torch.zeros((len(split.hidden_node_ids), *split.observed_data.y.shape[1:]), dtype=split.observed_data.y.dtype)
                        augmented.y = torch.cat([split.observed_data.y.cpu(), gen_pad], dim=0)
                        aug_train = torch.cat([train_mask, torch.zeros(len(split.hidden_node_ids), dtype=torch.bool)])
                        aug_test = torch.cat([test_mask, torch.zeros(len(split.hidden_node_ids), dtype=torch.bool)])
                        aug_metrics = train_and_evaluate_node_classifier(augmented, aug_train, aug_test, cfg)
                    except Exception as exc:
                        row.update({"status": "not_run", "reason": str(exc)})
                        rows.append(row)
                        continue
                row["status"] = aug_metrics.get("status", "ok")
                for metric in ["accuracy", "macro_f1", "micro_f1", "macro_auroc", "micro_auroc"]:
                    row[f"observed_incomplete_{metric}"] = observed_metrics.get(metric, np.nan)
                    row[f"augmented_graph_{metric}"] = aug_metrics.get(metric, np.nan)
                    row[f"full_graph_reference_{metric}"] = full_metrics.get(metric, np.nan)
                    row[f"delta_to_full_{metric}"] = aug_metrics.get(metric, np.nan) - full_metrics.get(metric, np.nan)
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw_dir / "downstream_classification.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_dir / "downstream_classification.csv", index=False)
    numeric = raw.select_dtypes(include=[np.number]).columns
    if not raw.empty and {"status", "dataset", "model"}.issubset(raw.columns):
        ok = raw[raw["status"] == "ok"]
        if not ok.empty:
            summary = ok.groupby(["dataset", "model"])[numeric].agg(["mean", "std"]).reset_index()
            summary.to_csv(summary_dir / "downstream_robustness.csv", index=False)
        else:
            pd.DataFrame(columns=["dataset", "model", "status"]).to_csv(summary_dir / "downstream_robustness.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
