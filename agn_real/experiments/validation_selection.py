from __future__ import annotations

import argparse
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from agn_real.data import load_real_dataset
from agn_real.eval.metrics import evaluate_all
from agn_real.experiments.run_real_node_holdout import run_rc_agn
from agn_real.splits import stratified_node_holdout
from agn_real.utils import apply_feature_mode, ensure_dir


@dataclass
class SelectionConfig:
    feature_mode: str = "hybrid_feature"
    k: int = 10
    tau: float = 0.0
    latent_dim: int = 32
    feature_mmd_weight: float = 0.5
    centroid_weight: float = 0.5
    interpolation_lambda: float = 0.5
    alpha: float = 0.6
    beta: float = 0.2
    gamma: float = 0.1
    delta: float = 0.1


def validation_score(metrics: Dict[str, float], precision_weight: float = 1.0, ap_weight: float = 1.0, mmd_weight: float = 1.0) -> float:
    precision = float(metrics.get("hidden_edge_precision_at_k", 0.0) or 0.0)
    ap = float(metrics.get("attachment_ap_comparable", 0.0) or 0.0)
    mmd = float(metrics.get("feature_mmd", 0.0) or 0.0)
    return precision_weight * precision + ap_weight * ap - mmd_weight * mmd


def small_selection_grid(grid: str = "quick") -> Iterable[SelectionConfig]:
    if grid == "quick":
        yield SelectionConfig(feature_mode="decoder", feature_mmd_weight=0.0, centroid_weight=0.0, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
        yield SelectionConfig(feature_mode="decoder_mmd", feature_mmd_weight=0.5, centroid_weight=0.0, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
        yield SelectionConfig(feature_mode="role_interpolation", feature_mmd_weight=0.0, centroid_weight=0.0, alpha=0.7, beta=0.2, gamma=0.0, delta=0.1)
        yield SelectionConfig(feature_mode="hybrid_feature", feature_mmd_weight=0.5, centroid_weight=0.5, alpha=0.6, beta=0.2, gamma=0.1, delta=0.1)
        return
    feature_modes = ["decoder", "decoder_mmd", "centroid_regularized", "role_interpolation", "hybrid_feature"]
    for feature_mode in feature_modes:
        for beta in [0.0, 0.2, 0.4]:
            yield SelectionConfig(
                feature_mode=feature_mode,
                feature_mmd_weight=0.5 if feature_mode in {"decoder_mmd", "hybrid_feature"} else 0.0,
                centroid_weight=0.5 if feature_mode in {"centroid_regularized", "hybrid_feature"} else 0.0,
                beta=beta,
                alpha=max(0.1, 0.8 - beta),
                gamma=0.1,
                delta=0.1 if feature_mode in {"role_interpolation", "hybrid_feature"} else 0.0,
            )


def _apply_selection_args(args: argparse.Namespace, cfg: SelectionConfig) -> argparse.Namespace:
    tuned = copy(args)
    tuned.agn_feature_mode = cfg.feature_mode
    tuned.agn_attachment_mode = "hybrid"
    tuned.k = cfg.k
    tuned.attachment_threshold = cfg.tau
    tuned.latent_dim = cfg.latent_dim
    tuned.feature_mmd_weight = cfg.feature_mmd_weight
    tuned.feature_centroid_weight = cfg.centroid_weight
    tuned.interpolation_lambda = cfg.interpolation_lambda
    tuned.hybrid_alpha = cfg.alpha
    tuned.hybrid_beta = cfg.beta
    tuned.hybrid_gamma = cfg.gamma
    tuned.hybrid_delta = cfg.delta
    return tuned


def select_on_observed_validation(observed_data, seed: int, args: argparse.Namespace) -> Tuple[SelectionConfig, Dict[str, float], List[Dict[str, object]]]:
    validation_split = stratified_node_holdout(observed_data, getattr(args, "validation_holdout_ratio", 0.10), seed + 1009, "degree_stratified")
    rows: List[Dict[str, object]] = []
    best_cfg = None
    best_metrics: Dict[str, float] = {}
    best_score = float("-inf")
    for cfg in small_selection_grid(getattr(args, "selection_grid", "quick")):
        tuned = _apply_selection_args(args, cfg)
        tuned.epochs = min(getattr(args, "epochs", 10), getattr(args, "validation_epochs", 5))
        tuned.attachment_epochs = min(getattr(args, "attachment_epochs", 10), getattr(args, "validation_attachment_epochs", 5))
        result = run_rc_agn(validation_split.observed_data, len(validation_split.hidden_node_ids), seed, tuned)
        metrics = evaluate_all(validation_split, result["generated_x"], result["edge_index"])
        score = validation_score(
            metrics,
            precision_weight=getattr(args, "validation_precision_weight", 1.0),
            ap_weight=getattr(args, "validation_ap_weight", 1.0),
            mmd_weight=getattr(args, "validation_mmd_weight", 1.0),
        )
        row = {**asdict(cfg), **metrics, "validation_score": score}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics
    if best_cfg is None:
        best_cfg = SelectionConfig()
    return best_cfg, best_metrics, rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select improved AGN hyperparameters on an observed-only validation holdout.")
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default="results/final")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--holdout_ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--feature_mode", default="raw")
    parser.add_argument("--validation_holdout_ratio", type=float, default=0.10)
    parser.add_argument("--selection_grid", choices=["quick", "full"], default="quick")
    parser.add_argument("--validation_epochs", type=int, default=5)
    parser.add_argument("--validation_attachment_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--attachment_epochs", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--encoder", default="gcn", choices=["gcn", "sage", "gat"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--attachment_threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation_precision_weight", type=float, default=1.0)
    parser.add_argument("--validation_ap_weight", type=float, default=1.0)
    parser.add_argument("--validation_mmd_weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.dataset_root:
        args.data_root = args.dataset_root
    rows = []
    summary_dir = ensure_dir(Path(args.output_dir) / "summary")
    out_path = summary_dir / "validation_selection.csv"
    for dataset in args.datasets:
        data, _ = load_real_dataset(dataset, args.data_root)
        data = apply_feature_mode(data, args.feature_mode)
        for ratio in args.holdout_ratios:
            for seed in args.seeds:
                test_split = stratified_node_holdout(data, ratio, seed, "degree_stratified")
                best_cfg, best_val_metrics, _ = select_on_observed_validation(test_split.observed_data, seed, args)
                tuned = _apply_selection_args(args, best_cfg)
                result = run_rc_agn(test_split.observed_data, len(test_split.hidden_node_ids), seed, tuned)
                test_metrics = evaluate_all(test_split, result["generated_x"], result["edge_index"])
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "holdout_ratio": ratio,
                        "selected_feature_mode": best_cfg.feature_mode,
                        "selected_k": best_cfg.k,
                        "selected_tau": best_cfg.tau,
                        "selected_latent_dim": best_cfg.latent_dim,
                        "selected_feature_mmd_weight": best_cfg.feature_mmd_weight,
                        "selected_centroid_weight": best_cfg.centroid_weight,
                        "selected_alpha": best_cfg.alpha,
                        "selected_beta": best_cfg.beta,
                        "selected_gamma": best_cfg.gamma,
                        "selected_delta": best_cfg.delta,
                        "validation_metric": validation_score(best_val_metrics),
                        "test_metric": validation_score(test_metrics),
                        "test_feature_mmd": test_metrics.get("feature_mmd"),
                        "test_attachment_precision_at_k": test_metrics.get("hidden_edge_precision_at_k"),
                        "test_attachment_ap_comparable": test_metrics.get("attachment_ap_comparable"),
                    }
                )
                pd.DataFrame(rows).to_csv(out_path, index=False)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
