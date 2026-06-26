from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agn_real.utils import ensure_dir
from agn_real.reporting.model_names import apply_model_display_names


def _bar(df: pd.DataFrame, metric: str, path: Path) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(data=apply_model_display_names(df[df["status"] == "ok"]), x="dataset", y=metric, hue="model", errorbar="sd")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build AGN real-data figures.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args(argv)
    root = Path(args.results_dir)
    fig_dir = ensure_dir(root / "figures")
    raw_path = root / "raw" / "real_node_holdout_results.csv"
    if not raw_path.exists():
        return 0
    df = apply_model_display_names(pd.read_csv(raw_path))
    _bar(df, "density_error", fig_dir / "topology_density_error.png")
    _bar(df, "average_degree_error", fig_dir / "topology_average_degree_error.png")
    _bar(df, "feature_mmd", fig_dir / "hidden_node_feature_mmd.png")
    _bar(df, "hidden_edge_precision_at_k", fig_dir / "attachment_precision_at_k.png")
    _bar(df, "runtime_sec", fig_dir / "scalability_runtime.png")
    full = df[df["dataset"].isin(["Cora", "CiteSeer"])].copy()
    _bar(full, "hidden_edge_precision_at_k", fig_dir / "main_attachment_precision_at_k.pdf")
    _bar(full, "feature_mmd", fig_dir / "main_feature_mmd.pdf")
    if {"attachment_auc", "attachment_ap"}.intersection(full.columns):
        _bar(full, "attachment_ap", fig_dir / "main_auc_ap.pdf")
    down_path = root / "raw" / "downstream_classification.csv"
    if down_path.exists():
        down = apply_model_display_names(pd.read_csv(down_path))
        if "delta_to_full_macro_f1" in down.columns:
            plt.figure(figsize=(10, 5))
            plot_df = down[down["status"].eq("ok")].copy()
            plot_df["abs_delta_to_full_macro_f1"] = plot_df["delta_to_full_macro_f1"].abs()
            sns.barplot(data=plot_df, x="dataset", y="abs_delta_to_full_macro_f1", hue="model", errorbar="sd")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(fig_dir / "downstream_robustness.pdf")
            plt.close()
    sens = root / "summary" / "sensitivity_grid.csv"
    if sens.exists():
        s = pd.read_csv(sens)
        if {"k", "latent_dim", "attachment_threshold"}.issubset(s.columns):
            plt.figure(figsize=(7, 5))
            pivot = s.pivot_table(index="k", columns="latent_dim", values="attachment_threshold", aggfunc="mean")
            sns.heatmap(pivot, annot=True, fmt=".2f")
            ensure_dir(fig_dir / "sensitivity_heatmaps")
            plt.tight_layout()
            plt.savefig(fig_dir / "sensitivity_heatmaps" / "grid_coverage.png", dpi=200)
            plt.close()
    curves = root / "summary" / "sensitivity_curves.csv"
    if curves.exists():
        c = pd.read_csv(curves)
        for param, filename in [
            ("k", "sensitivity_k_curve.pdf"),
            ("tau", "sensitivity_tau_curve.pdf"),
            ("holdout_ratio", "sensitivity_holdout_ratio_curve.pdf"),
            ("latent_dim", "sensitivity_latent_dim_curve.pdf"),
        ]:
            sub = c[(c.get("varied_parameter") == param) & (c.get("metric") == "density_error")].dropna(subset=["mean"])
            if sub.empty:
                continue
            plt.figure(figsize=(6, 4))
            sub = sub.assign(parameter_value_float=sub["parameter_value"].astype(float)).sort_values("parameter_value_float")
            plt.errorbar(sub["parameter_value_float"], sub["mean"], yerr=sub["ci95"].fillna(sub["std"]), marker="o", capsize=3)
            plt.xlabel(param)
            plt.ylabel("Density error, mean +/- 95% CI")
            plt.tight_layout()
            plt.savefig(fig_dir / filename)
            plt.close()
        for metric, filename in [
            ("hidden_edge_precision_at_k", "sensitivity_attachment_precision.pdf"),
            ("feature_mmd", "sensitivity_feature_mmd_curve.pdf"),
            ("attachment_ap", "sensitivity_auc_ap_curve.pdf"),
        ]:
            sub = c[(c.get("metric") == metric) & (c.get("varied_parameter") == "k")].dropna(subset=["mean"])
            if sub.empty:
                continue
            plt.figure(figsize=(6, 4))
            sub = sub.assign(parameter_value_float=sub["parameter_value"].astype(float)).sort_values("parameter_value_float")
            plt.errorbar(sub["parameter_value_float"], sub["mean"], yerr=sub["ci95"].fillna(sub["std"]), marker="o", capsize=3)
            plt.xlabel("k")
            plt.ylabel(f"{metric}, mean +/- 95% CI")
            plt.tight_layout()
            plt.savefig(fig_dir / filename)
            plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
