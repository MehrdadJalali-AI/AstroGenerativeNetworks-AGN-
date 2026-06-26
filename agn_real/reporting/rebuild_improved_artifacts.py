from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agn_real.reporting.model_names import MAIN_MODEL_ORDER, apply_model_display_names
from agn_real.utils import ensure_dir


def _note_for_dataset(dataset: str) -> str:
    if dataset in {"Cora", "CiteSeer", "PPI"}:
        return "full 5-seed result with validation-selected improved AGN"
    if dataset == "PubMed":
        return "partial/preliminary"
    if dataset == "AmazonComputers":
        return "not_run due to runtime"
    if dataset == "CoauthorCS":
        return "not_run due to download failure"
    return ""


def _manuscript_table_2(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = apply_model_display_names(raw[raw["status"].eq("ok")]).copy()
    metrics = [
        "feature_mmd",
        "hidden_edge_precision_at_k",
        "hidden_edge_recall_at_k",
        "attachment_auc_comparable",
        "attachment_ap_comparable",
        "density_error",
        "average_degree_error",
        "role_distribution_js",
    ]
    out = df.groupby(["dataset", "holdout_ratio", "model"], dropna=False)[[m for m in metrics if m in df.columns]].agg(["mean", "std", "count"]).reset_index()
    out.columns = ["_".join([str(c) for c in col if c != ""]).rstrip("_") for col in out.columns]
    out["note"] = out["dataset"].map(_note_for_dataset)
    order = {m: i for i, m in enumerate(MAIN_MODEL_ORDER)}
    out["_order"] = out["model"].map(order).fillna(99)
    out = out.sort_values(["dataset", "holdout_ratio", "_order"]).drop(columns="_order")
    out.to_csv(path, index=False)
    return out


def _manuscript_table_3(down: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = apply_model_display_names(down[down["status"].eq("ok")]).copy()
    numeric = [c for c in df.select_dtypes("number").columns if c not in {"seed"}]
    out = df.groupby(["dataset", "model"], dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([str(c) for c in col if c != ""]).rstrip("_") for col in out.columns]
    out["note"] = out["dataset"].map(lambda d: "full 5-seed downstream result" if d in {"Cora", "CiteSeer", "PPI"} else _note_for_dataset(d))
    order = {m: i for i, m in enumerate(["Observed incomplete", *MAIN_MODEL_ORDER])}
    out["_order"] = out["model"].map(order).fillna(99)
    out = out.sort_values(["dataset", "_order"]).drop(columns="_order")
    out.to_csv(path, index=False)
    return out


def _plot_metric(df: pd.DataFrame, metric: str, path: Path, lower_is_better: bool = False) -> None:
    if metric not in df.columns:
        return
    plot_df = apply_model_display_names(df[df["status"].eq("ok") & df["dataset"].isin(["Cora", "CiteSeer", "PPI"])]).copy()
    plot_df = plot_df[plot_df["model"].isin(MAIN_MODEL_ORDER)]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="dataset", y=metric, hue="model", errorbar="sd", hue_order=MAIN_MODEL_ORDER)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric.replace("_", " "))
    if lower_is_better:
        plt.title("Lower is better")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_ablation(ablation: pd.DataFrame, fig_dir: Path) -> None:
    ok = ablation[ablation["status"].eq("ok")].copy()
    if ok.empty:
        return
    summary = ok.groupby("model_label")[["feature_mmd", "hidden_edge_precision_at_k"]].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary, x="feature_mmd", y="hidden_edge_precision_at_k", hue="model_label", s=90)
    plt.xlabel("Feature MMD (lower is better)")
    plt.ylabel("Attachment precision@k (higher is better)")
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "agn_ablation_feature_vs_attachment.pdf")
    plt.close()
    if {"agn_attachment_mode", "hidden_edge_precision_at_k"}.issubset(ok.columns):
        weights = ok[ok["agn_attachment_mode"].eq("hybrid")].copy()
        if not weights.empty:
            plt.figure(figsize=(7, 4))
            sns.barplot(data=weights, x="model_label", y="hidden_edge_precision_at_k", errorbar="sd")
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Attachment precision@k")
            plt.tight_layout()
            plt.savefig(fig_dir / "hybrid_attachment_weight_sensitivity.pdf")
            plt.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge improved AGN runs and rebuild final manuscript artifacts.")
    parser.add_argument("--results_dir", default="results/final")
    parser.add_argument("--improved_main_dir", default="results/final_improved_main2")
    parser.add_argument("--improved_downstream_dir", default="results/final_improved_downstream")
    args = parser.parse_args(argv)

    root = Path(args.results_dir)
    raw_dir = ensure_dir(root / "raw")
    summary_dir = ensure_dir(root / "summary")
    table_dir = ensure_dir(root / "tables")
    fig_dir = ensure_dir(root / "figures")

    old_raw = pd.read_csv(raw_dir / "real_node_holdout_results.csv")
    improved = pd.read_csv(Path(args.improved_main_dir) / "raw" / "real_node_holdout_results.csv")
    keep = old_raw[~old_raw["dataset"].isin(["Cora", "CiteSeer", "PPI"])].copy()
    raw = pd.concat([improved, keep], ignore_index=True, sort=False)
    raw.to_csv(raw_dir / "real_node_holdout_results.csv", index=False)

    old_down = pd.read_csv(raw_dir / "downstream_classification.csv")
    improved_down = pd.read_csv(Path(args.improved_downstream_dir) / "raw" / "downstream_classification.csv")
    keep_down = old_down[~old_down["dataset"].isin(["Cora", "CiteSeer", "PPI"])].copy()
    down = pd.concat([improved_down, keep_down], ignore_index=True, sort=False)
    down.to_csv(raw_dir / "downstream_classification.csv", index=False)

    numeric = raw.select_dtypes("number").columns.tolist()
    group_cols = ["dataset", "holdout_ratio", "feature_mode", "model"]
    summary = raw[raw["status"].eq("ok")].groupby(group_cols, dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c != ""]).rstrip("_") for col in summary.columns]
    summary.to_csv(summary_dir / "main_real_node_holdout_results.csv", index=False)
    down_numeric = down.select_dtypes("number").columns.tolist()
    down_summary = down[down["status"].eq("ok")].groupby(["dataset", "model"], dropna=False)[down_numeric].agg(["mean", "std"]).reset_index()
    down_summary.columns = ["_".join([str(c) for c in col if c != ""]).rstrip("_") for col in down_summary.columns]
    down_summary.to_csv(summary_dir / "downstream_robustness.csv", index=False)

    if (summary_dir / "dataset_summary.csv").exists():
        ds = pd.read_csv(summary_dir / "dataset_summary.csv")
        ds["note"] = ds["dataset"].map(_note_for_dataset)
        ds.to_csv(table_dir / "MANUSCRIPT_Table_1_dataset_summary.csv", index=False)
    _manuscript_table_2(raw, table_dir / "MANUSCRIPT_Table_2_main_node_holdout.csv")
    _manuscript_table_3(down, table_dir / "MANUSCRIPT_Table_3_downstream_classification.csv")
    if (summary_dir / "sensitivity_curves.csv").exists():
        sens = pd.read_csv(summary_dir / "sensitivity_curves.csv")
        sens["note"] = "pre-improvement sensitivity result"
        sens.to_csv(table_dir / "MANUSCRIPT_Table_4_sensitivity.csv", index=False)
    if (summary_dir / "scalability.csv").exists():
        sc = pd.read_csv(summary_dir / "scalability.csv")
        sc["note"] = sc["dataset"].map(_note_for_dataset).fillna("")
        not_run = raw[raw["status"].eq("not_run")][["dataset", "status", "reason"]].drop_duplicates("dataset")
        if not not_run.empty:
            not_run["note"] = not_run["dataset"].map(_note_for_dataset)
            for col in sc.columns:
                if col not in not_run.columns:
                    not_run[col] = pd.NA
            for col in not_run.columns:
                if col not in sc.columns:
                    sc[col] = pd.NA
            sc = pd.concat([sc, not_run[sc.columns]], ignore_index=True, sort=False)
        sc.to_csv(table_dir / "MANUSCRIPT_Table_5_scalability_and_notrun.csv", index=False)

    _plot_metric(raw, "hidden_edge_precision_at_k", fig_dir / "main_attachment_precision_at_k.pdf")
    _plot_metric(raw, "feature_mmd", fig_dir / "main_feature_mmd.pdf", lower_is_better=True)
    _plot_metric(raw, "attachment_auc_comparable", fig_dir / "main_attachment_auc_comparable.pdf")
    _plot_metric(raw, "attachment_ap_comparable", fig_dir / "main_attachment_ap_comparable.pdf")
    if "delta_to_full_macro_f1" in down.columns:
        plot = apply_model_display_names(down[down["status"].eq("ok")]).copy()
        plot["abs_delta_to_full_macro_f1"] = plot["delta_to_full_macro_f1"].abs()
        plt.figure(figsize=(10, 5))
        sns.barplot(data=plot, x="dataset", y="abs_delta_to_full_macro_f1", hue="model", errorbar="sd")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("|Delta to full graph macro-F1|")
        plt.tight_layout()
        plt.savefig(fig_dir / "downstream_robustness.pdf")
        plt.close()

    ablation_path = raw_dir / "agn_improvement_ablation_raw.csv"
    if ablation_path.exists():
        _plot_ablation(pd.read_csv(ablation_path), fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
