from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats

from agn_real.baselines import run_baseline
from agn_real.data import load_real_dataset, save_dataset_summary_csv
from agn_real.eval.metrics import evaluate_all
from agn_real.models.attachment_ranker import attach_generated_nodes, train_attachment_ranker
from agn_real.models.feature_generator import feature_mode_uses_interpolation, role_interpolated_features
from agn_real.models.hybrid_attachment import HybridAttachmentConfig, attach_hybrid_generated_nodes
from agn_real.models.rc_agn import RCAGNConfig, compute_role_ids, generate_nodes, train_model
from agn_real.splits import stratified_node_holdout
from agn_real.utils import apply_feature_mode, ensure_dir, set_seed


DEFAULT_DATASETS = ["Cora", "CiteSeer", "PubMed", "PPI", "CoauthorCS", "CoauthorPhysics", "AmazonComputers", "AmazonPhoto"]
DEFAULT_MODELS = ["agn", "unconditioned_cosine", "fukushima_yamanishi_gca", "vgae", "knn_raw", "preferential", "random"]


def _role_distribution(roles: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.bincount(roles.cpu(), minlength=bins).float()


def run_rc_agn(observed_data, num_nodes: int, seed: int, args) -> Dict[str, object]:
    agn_feature_mode = getattr(args, "agn_feature_mode", getattr(args, "feature_generator_mode", "hybrid_feature"))
    cfg = RCAGNConfig(
        input_dim=observed_data.x.size(1),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        encoder_type=args.encoder,
        epochs=args.epochs,
        feature_mmd_weight=getattr(args, "feature_mmd_weight", 0.0),
        feature_centroid_weight=getattr(args, "feature_centroid_weight", 0.0),
        feature_mode=agn_feature_mode,
        interpolation_lambda=getattr(args, "interpolation_lambda", 0.5),
        conditional_prior=True,
        seed=seed,
        device=args.device,
    )
    model, train_info = train_model(observed_data, cfg)
    roles = compute_role_ids(observed_data, cfg.num_role_bins)
    gen = generate_nodes(model, num_nodes, _role_distribution(roles, cfg.num_role_bins), cfg)
    if feature_mode_uses_interpolation(agn_feature_mode):
        gen["x"] = role_interpolated_features(gen["x"], gen["roles"], observed_data, roles, cfg.interpolation_lambda, seed)
    ranker, ranker_info = train_attachment_ranker(observed_data, node_z=None, config={"epochs": max(5, args.attachment_epochs), "seed": seed, "device": args.device})
    attachment_mode = getattr(args, "agn_attachment_mode", "hybrid")
    if attachment_mode == "learned_ranker":
        edges = attach_generated_nodes(observed_data, gen["x"], gen["z"], ranker=ranker, mode="learned_ranker", k=args.k, threshold=args.attachment_threshold, seed=seed)
        edge_scores = None
    else:
        hybrid_cfg = HybridAttachmentConfig(
            alpha=getattr(args, "hybrid_alpha", 0.6),
            beta=getattr(args, "hybrid_beta", 0.2),
            gamma=getattr(args, "hybrid_gamma", 0.1),
            delta=getattr(args, "hybrid_delta", 0.1),
            k=args.k,
            threshold=args.attachment_threshold,
        )
        edges, edge_scores = attach_hybrid_generated_nodes(observed_data, gen["x"], gen["z"], ranker, hybrid_cfg, gen.get("roles"))
        ranker_info.update({"hybrid_alpha": hybrid_cfg.alpha, "hybrid_beta": hybrid_cfg.beta, "hybrid_gamma": hybrid_cfg.gamma, "hybrid_delta": hybrid_cfg.delta})
    return {
        "status": "ok",
        "generated_x": gen["x"],
        "generated_z": gen["z"],
        "edge_index": edges,
        "edge_scores": edge_scores,
        "train_info": train_info,
        "attachment_info": ranker_info,
        "agn_feature_mode": agn_feature_mode,
        "agn_attachment_mode": attachment_mode,
    }


def run_rc_agn_ablation(model_name: str, observed_data, num_nodes: int, seed: int, args) -> Dict[str, object]:
    original = {
        "agn_feature_mode": getattr(args, "agn_feature_mode", None),
        "agn_attachment_mode": getattr(args, "agn_attachment_mode", None),
        "feature_mmd_weight": getattr(args, "feature_mmd_weight", None),
        "feature_centroid_weight": getattr(args, "feature_centroid_weight", None),
        "hybrid_alpha": getattr(args, "hybrid_alpha", None),
        "hybrid_beta": getattr(args, "hybrid_beta", None),
        "hybrid_gamma": getattr(args, "hybrid_gamma", None),
        "hybrid_delta": getattr(args, "hybrid_delta", None),
    }
    name = model_name.lower()
    if name in {"agn_current", "agn_without_feature_mmd_loss", "agn_without_preferential_blending", "agn_without_validation_selection"}:
        args.agn_feature_mode = "decoder"
        args.agn_attachment_mode = "learned_ranker"
        args.feature_mmd_weight = 0.0
        args.feature_centroid_weight = 0.0
    elif name == "agn_preferential_blend":
        args.agn_feature_mode = "decoder"
        args.agn_attachment_mode = "hybrid"
        args.feature_mmd_weight = 0.0
        args.feature_centroid_weight = 0.0
        args.hybrid_alpha, args.hybrid_beta, args.hybrid_gamma, args.hybrid_delta = 0.7, 0.3, 0.0, 0.0
    elif name == "agn_feature_mmd":
        args.agn_feature_mode = "decoder_mmd"
        args.agn_attachment_mode = "learned_ranker"
        args.feature_mmd_weight = max(float(getattr(args, "feature_mmd_weight", 0.0)), 1.0)
        args.feature_centroid_weight = 0.0
    elif name == "agn_feature_centroid":
        args.agn_feature_mode = "centroid_regularized"
        args.agn_attachment_mode = "learned_ranker"
        args.feature_mmd_weight = 0.0
        args.feature_centroid_weight = max(float(getattr(args, "feature_centroid_weight", 0.0)), 1.0)
    elif name == "agn_role_interpolation":
        args.agn_feature_mode = "role_interpolation"
        args.agn_attachment_mode = "learned_ranker"
        args.feature_mmd_weight = 0.0
        args.feature_centroid_weight = 0.0
    elif name == "agn_mmd_preferential":
        args.agn_feature_mode = "decoder_mmd"
        args.agn_attachment_mode = "hybrid"
        args.feature_mmd_weight = max(float(getattr(args, "feature_mmd_weight", 0.0)), 1.0)
        args.feature_centroid_weight = 0.0
        args.hybrid_alpha, args.hybrid_beta, args.hybrid_gamma, args.hybrid_delta = 0.7, 0.3, 0.0, 0.0
    elif name == "agn_mmd_centroid_preferential":
        args.agn_feature_mode = "centroid_regularized"
        args.agn_attachment_mode = "hybrid"
        args.feature_mmd_weight = max(float(getattr(args, "feature_mmd_weight", 0.0)), 0.5)
        args.feature_centroid_weight = max(float(getattr(args, "feature_centroid_weight", 0.0)), 0.5)
        args.hybrid_alpha, args.hybrid_beta, args.hybrid_gamma, args.hybrid_delta = 0.6, 0.3, 0.1, 0.0
    elif name in {"agn_hybrid_feature_hybrid_attachment_validation", "agn"}:
        args.agn_feature_mode = "hybrid_feature"
        args.agn_attachment_mode = "hybrid"
        args.feature_mmd_weight = max(float(getattr(args, "feature_mmd_weight", 0.0)), 0.5)
        args.feature_centroid_weight = max(float(getattr(args, "feature_centroid_weight", 0.0)), 0.5)
    else:
        raise ValueError(f"Unknown AGN ablation '{model_name}'")
    try:
        result = run_rc_agn(observed_data, num_nodes, seed, args)
        result["agn_variant"] = model_name
        return result
    finally:
        for key, value in original.items():
            if value is None and hasattr(args, key):
                delattr(args, key)
            elif value is not None:
                setattr(args, key, value)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = [c for c in ["dataset", "holdout_ratio", "feature_mode", "model"] if c in raw.columns]
    rows = []
    for keys, group in raw[raw["status"] == "ok"].groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in numeric_cols:
            if metric in {"seed", "runtime_sec"} or metric in group_cols:
                continue
            values = group[metric].dropna().astype(float)
            if values.empty:
                continue
            row = {**base, "metric": metric, "n": len(values), "mean": values.mean(), "std": values.std(ddof=1) if len(values) > 1 else 0.0}
            row["ci95"] = 1.96 * row["std"] / np.sqrt(max(1, row["n"]))
            rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty and "model" in raw.columns:
        tests = []
        for (dataset, ratio, feature_mode, metric), group in raw[raw["status"] == "ok"].melt(
            id_vars=["dataset", "holdout_ratio", "feature_mode", "model", "seed", "status"],
            value_vars=[c for c in numeric_cols if c not in {"seed", "runtime_sec"}],
            var_name="metric",
            value_name="value",
        ).dropna().groupby(["dataset", "holdout_ratio", "feature_mode", "metric"]):
            pivot = group.pivot_table(index="seed", columns="model", values="value", aggfunc="first")
            if "rc_agn" not in pivot.columns:
                continue
            for model in pivot.columns:
                if model == "rc_agn":
                    continue
                paired = pivot[["rc_agn", model]].dropna()
                if len(paired) < 2:
                    continue
                diff = paired["rc_agn"] - paired[model]
                try:
                    stat, p = stats.wilcoxon(paired["rc_agn"], paired[model])
                    test = "wilcoxon"
                except Exception:
                    stat, p = stats.ttest_rel(paired["rc_agn"], paired[model])
                    test = "paired_t"
                tests.append({"dataset": dataset, "holdout_ratio": ratio, "feature_mode": feature_mode, "metric": metric, "baseline": model, "test": test, "p_value": p, "effect_size_mean_diff": diff.mean()})
        if tests:
            summary = summary.merge(pd.DataFrame(tests), on=["dataset", "holdout_ratio", "feature_mode", "metric"], how="left")
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real-data node-holdout AGN recovery experiments.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--dataset_root", default=None, help="Alias for --data_root.")
    parser.add_argument("--ppi_root", default=None)
    parser.add_argument("--ppi_mode", default="selected_graph", choices=["selected_graph", "disjoint_union"])
    parser.add_argument("--ppi_graph_index", type=int, default=0)
    parser.add_argument("--ppi_max_nodes", type=int, default=None)
    parser.add_argument("--holdout_ratios", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--feature_modes", nargs="+", default=["raw"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--split_strategy", default="degree_stratified")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--attachment_epochs", type=int, default=20)
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.dataset_root:
        args.data_root = args.dataset_root

    out = Path(args.output_dir)
    raw_dir = ensure_dir(out / "raw")
    summary_dir = ensure_dir(out / "summary")
    rows = []
    summaries = []
    for dataset in args.datasets:
        try:
            base_data, meta = load_real_dataset(
                dataset,
                args.data_root,
                ppi_root=args.ppi_root,
                ppi_mode=args.ppi_mode,
                ppi_graph_index=args.ppi_graph_index,
                ppi_max_nodes=args.ppi_max_nodes,
            )
            summaries.append(meta)
        except Exception as exc:
            for model in args.models:
                rows.append({"dataset": dataset, "model": model, "status": "not_run", "reason": f"dataset load failed: {exc}"})
            continue
        for feature_mode in args.feature_modes:
            try:
                data = apply_feature_mode(base_data, feature_mode)
            except Exception as exc:
                rows.append({"dataset": dataset, "feature_mode": feature_mode, "status": "not_run", "reason": str(exc)})
                continue
            for ratio in args.holdout_ratios:
                for seed in args.seeds:
                    set_seed(seed)
                    try:
                        split = stratified_node_holdout(data, ratio, seed, args.split_strategy)
                    except Exception as exc:
                        rows.append({"dataset": dataset, "feature_mode": feature_mode, "holdout_ratio": ratio, "seed": seed, "status": "not_run", "reason": f"split failed: {exc}"})
                        continue
                    for model_name in args.models:
                        start = time.perf_counter()
                        row = {"dataset": dataset, "feature_mode": feature_mode, "holdout_ratio": ratio, "seed": seed, "model": model_name}
                        try:
                            if model_name in {"rc_agn", "agn"}:
                                result = run_rc_agn(split.observed_data, len(split.hidden_node_ids), seed, args)
                            elif model_name.startswith("agn_"):
                                result = run_rc_agn_ablation(model_name, split.observed_data, len(split.hidden_node_ids), seed, args)
                            else:
                                result = run_baseline(model_name, split.observed_data, len(split.hidden_node_ids), seed=seed, k=args.k, threshold=args.attachment_threshold, epochs=args.epochs)
                            row["status"] = result["status"]
                            if result["status"] == "ok":
                                row.update(evaluate_all(split, result["generated_x"], result["edge_index"]))
                                row["agn_feature_mode"] = result.get("agn_feature_mode")
                                row["agn_attachment_mode"] = result.get("agn_attachment_mode")
                                row["agn_variant"] = result.get("agn_variant")
                                attach_info = result.get("attachment_info", {})
                                if "train_auc" in attach_info:
                                    row["attachment_auc"] = attach_info["train_auc"]
                                if "train_ap" in attach_info:
                                    row["attachment_ap"] = attach_info["train_ap"]
                            else:
                                row["reason"] = result.get("reason", "unavailable")
                        except Exception as exc:
                            row["status"] = "not_run"
                            row["reason"] = str(exc)
                        row["runtime_sec"] = time.perf_counter() - start
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(raw_dir / "real_node_holdout_results.csv", index=False)
    if summaries:
        save_dataset_summary_csv(summaries, summary_dir / "dataset_summary.csv")
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_dir / "real_node_holdout_results.csv", index=False)
    summarize(raw).to_csv(summary_dir / "main_real_node_holdout_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
