from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from agn_real.utils import ensure_dir
from agn_real.reporting.model_names import apply_model_display_names


TABLE_MAP = {
    "main_real_node_holdout_results.csv": "main_real_node_holdout_results.csv",
    "dataset_summary.csv": "dataset_summary.csv",
    "sensitivity_grid.csv": "sensitivity_grid.csv",
    "scalability.csv": "scalability.csv",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build publication-ready CSV tables from real-data AGN results.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args(argv)
    root = Path(args.results_dir)
    summary = root / "summary"
    tables = ensure_dir(root / "tables")
    for source, target in TABLE_MAP.items():
        src = summary / source
        if src.exists():
            try:
                apply_model_display_names(pd.read_csv(src)).to_csv(tables / target, index=False)
            except EmptyDataError:
                pd.DataFrame().to_csv(tables / target, index=False)
    dataset_summary = summary / "dataset_summary.csv"
    if dataset_summary.exists():
        ds = pd.read_csv(dataset_summary)
        rename = {
            "num_nodes": "nodes",
            "num_edges": "edges",
            "feature_dim": "feature dimension",
            "num_classes": "labels/classes",
            "average_degree": "average degree",
        }
        cols = ["dataset", "domain", "num_nodes", "num_edges", "feature_dim", "num_classes", "density", "average_degree"]
        ds[[c for c in cols if c in ds.columns]].rename(columns=rename).to_csv(tables / "table_A_real_dataset_summary.csv", index=False)
    raw = root / "raw" / "real_node_holdout_results.csv"
    if raw.exists():
        df = apply_model_display_names(pd.read_csv(raw))
        ok = df[df.get("status", "") == "ok"].copy()
        if not ok.empty:
            cols = [c for c in ["dataset", "holdout_ratio", "feature_mode", "model", "density_error", "average_degree_error", "feature_mmd", "degree_distribution_js", "hidden_edge_precision_at_k", "community_nmi_observed_nodes"] if c in ok.columns]
            ok[cols].to_csv(tables / "baseline_comparison.csv", index=False)
            if "model" in ok.columns:
                ok[ok["model"].isin(["AGN", "Unconditioned + cosine", "VGAE/GAE"])][cols].to_csv(tables / "ablation_role_conditioning.csv", index=False)
                ok[ok["model"].isin(["AGN", "Random", "Preferential", "Unconditioned + cosine", "GraphSAGE + learned attachment"])][cols].to_csv(tables / "ablation_attachment_strategy.csv", index=False)
            table_b_cols = [c for c in ["dataset", "model", "density_error", "average_degree_error", "feature_mmd", "attachment_auc", "attachment_ap", "role_distribution_js"] if c in ok.columns]
            ok[table_b_cols].rename(columns={
                "density_error": "topology recovery error",
                "feature_mmd": "feature distribution distance",
                "role_distribution_js": "role distribution JS divergence",
            }).to_csv(tables / "table_B_node_holdout_recovery.csv", index=False)
            metric = "density_error" if "density_error" in ok.columns else (ok.select_dtypes(include="number").columns[0] if len(ok.select_dtypes(include="number").columns) else None)
            if metric:
                rows = []
                for (dataset, m), group in ok.groupby(["dataset", "model"]):
                    vals = group[metric].dropna()
                    rows.append({"dataset": dataset, "model": m, "mean": vals.mean(), "std": vals.std(ddof=1) if len(vals) > 1 else 0.0})
                stats_df = pd.DataFrame(rows)
                for dataset, group in stats_df.groupby("dataset"):
                    if "AGN" not in set(group["model"]):
                        continue
                    agn = group[group["model"] == "AGN"].iloc[0]
                    baselines = group[group["model"] != "AGN"].sort_values("mean")
                    if baselines.empty:
                        continue
                    best = baselines.iloc[0]
                    pd.DataFrame([{
                        "dataset": dataset,
                        "metric": metric,
                        "AGN mean +/- std": f"{agn['mean']:.6g} +/- {agn['std']:.6g}",
                        "best baseline mean +/- std": f"{best['mean']:.6g} +/- {best['std']:.6g}",
                        "p-value": "",
                        "effect size": agn["mean"] - best["mean"],
                    }]).to_csv(tables / "table_D_multiseed_statistical_comparison.csv", mode="a", header=not (tables / "table_D_multiseed_statistical_comparison.csv").exists(), index=False)
    downstream = root / "raw" / "downstream_classification.csv"
    if downstream.exists():
        down = apply_model_display_names(pd.read_csv(downstream))
        cols = [c for c in [
            "dataset", "model",
            "observed_incomplete_accuracy", "augmented_graph_accuracy", "full_graph_reference_accuracy", "delta_to_full_accuracy",
            "observed_incomplete_macro_f1", "augmented_graph_macro_f1", "full_graph_reference_macro_f1", "delta_to_full_macro_f1",
            "observed_incomplete_micro_f1", "augmented_graph_micro_f1", "full_graph_reference_micro_f1", "delta_to_full_micro_f1",
        ] if c in down.columns]
        down[cols].to_csv(tables / "table_C_downstream_robustness.csv", index=False)
        down[cols].to_csv(tables / "downstream_robustness.csv", index=False)
    sensitivity = summary / "sensitivity_curves.csv"
    if sensitivity.exists():
        sens = pd.read_csv(sensitivity)
        rows = []
        for (dataset, param), group in sens.groupby([sens.get("dataset", pd.Series(["pooled"] * len(sens))), "varied_parameter"]):
            good = group.dropna(subset=["mean"]).sort_values("mean")
            if good.empty:
                continue
            best = good.iloc[0]
            stable = good[good["mean"] <= best["mean"] + good["std"].fillna(0).mean()]
            rows.append({
                "dataset": dataset,
                "varied parameter": param,
                "best value": best["parameter_value"],
                "stable range": ", ".join(map(str, stable["parameter_value"].tolist())),
                "main observation": f"lowest mean {best['metric']} in completed sensitivity runs",
            })
        pd.DataFrame(rows).to_csv(tables / "table_E_sensitivity_analysis_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
