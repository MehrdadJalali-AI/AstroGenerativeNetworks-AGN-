from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agn_real.experiments.run_real_node_holdout import main as run_main
from agn_real.utils import ensure_dir


def _collect_metrics(run_dir: Path, metrics: list[str] | None = None) -> list[dict]:
    metrics = metrics or ["hidden_edge_precision_at_k", "attachment_auc", "attachment_ap", "feature_mmd", "density_error"]
    path = run_dir / "raw" / "real_node_holdout_results.csv"
    if not path.exists():
        return [{"metric": m, "mean": np.nan, "std": np.nan, "ci95": np.nan, "n": 0} for m in metrics]
    df = pd.read_csv(path)
    rows = []
    for metric in metrics:
        if metric not in df.columns or "status" not in df.columns or "model" not in df.columns:
            rows.append({"metric": metric, "mean": np.nan, "std": np.nan, "ci95": np.nan, "n": 0})
            continue
        vals = df[(df.get("status") == "ok") & (df.get("model").isin(["rc_agn", "agn"]))][metric].dropna()
        if vals.empty:
            rows.append({"metric": metric, "mean": np.nan, "std": np.nan, "ci95": np.nan, "n": 0})
            continue
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append({"metric": metric, "mean": float(vals.mean()), "std": std, "ci95": 1.96 * std / np.sqrt(max(1, len(vals))), "n": int(len(vals))})
    return rows


def _plot_curve(df: pd.DataFrame, parameter: str, output: Path) -> None:
    sub = df[df["varied_parameter"].eq(parameter)].dropna(subset=["mean"])
    if sub.empty:
        return
    metric = sub["metric"].iloc[0]
    if sub["metric"].nunique() > 1:
        sub = sub[sub["metric"].eq(metric)]
    plt.figure(figsize=(6, 4))
    x = sub["parameter_value"].astype(float)
    y = sub["mean"].astype(float)
    err = sub["ci95"].fillna(sub["std"]).astype(float)
    order = np.argsort(x)
    plt.errorbar(x.iloc[order], y.iloc[order], yerr=err.iloc[order], marker="o", capsize=3)
    plt.xlabel(parameter)
    plt.ylabel(f"{metric}, mean +/- 95% CI")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run AGN sensitivity grid.")
    parser.add_argument("--datasets", nargs="+", default=["Cora"])
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--attachment_epochs", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Use a reduced grid for smoke testing.")
    args = parser.parse_args(argv)
    ks = [5, 10, 20, 30] if not args.quick else [5]
    thresholds = [0.3, 0.5, 0.7, 0.9] if not args.quick else [0.5]
    latent_dims = [16, 32, 64] if not args.quick else [16]
    ratios = [0.05, 0.10, 0.20] if not args.quick else [0.05]
    role_opts = [True, False] if not args.quick else [True]
    learned_opts = [True, False] if not args.quick else [True]
    rows = []
    base = {"k": 10, "attachment_threshold": 0.5, "latent_dim": 32, "hidden_node_ratio": 0.10}
    experiments = []
    experiments += [("k", k, {**base, "k": k}) for k in ks]
    experiments += [("tau", t, {**base, "attachment_threshold": t}) for t in thresholds]
    experiments += [("holdout_ratio", r, {**base, "hidden_node_ratio": r}) for r in ratios]
    experiments += [("latent_dim", z, {**base, "latent_dim": z}) for z in latent_dims]
    experiments += [("role_conditioning", v, base.copy()) for v in role_opts]
    experiments += [("learned_attachment", v, base.copy()) for v in learned_opts]
    for varied_parameter, parameter_value, cfg in experiments:
        models = ["rc_agn", "standard_agn"]
        if varied_parameter == "role_conditioning" and parameter_value is False:
            models = ["standard_agn"]
        if varied_parameter == "learned_attachment" and parameter_value is False:
            models = ["standard_agn"]
        run_dir = Path(args.output_dir) / "sensitivity_runs" / f"{varied_parameter}_{parameter_value}"
        run_main([
            "--datasets", *args.datasets,
            "--data_root", args.data_root,
            "--seeds", *map(str, args.seeds),
            "--holdout_ratios", str(cfg["hidden_node_ratio"]),
            "--feature_modes", "raw",
            "--models", *models,
            "--k", str(cfg["k"]),
            "--attachment_threshold", str(cfg["attachment_threshold"]),
            "--latent_dim", str(cfg["latent_dim"]),
            "--epochs", str(args.epochs),
            "--attachment_epochs", str(args.attachment_epochs),
            "--output_dir", str(run_dir),
        ])
        for metric_row in _collect_metrics(run_dir):
            rows.append({
                "varied_parameter": varied_parameter,
                "parameter_value": parameter_value,
                **cfg,
                "role_conditioning": not (varied_parameter == "role_conditioning" and parameter_value is False),
                "learned_attachment": not (varied_parameter == "learned_attachment" and parameter_value is False),
                "status": "completed",
                **metric_row,
            })
    out = ensure_dir(Path(args.output_dir) / "summary")
    curves = pd.DataFrame(rows)
    curves.to_csv(out / "sensitivity_curves.csv", index=False)
    curves.to_csv(out / "sensitivity_grid.csv", index=False)
    fig_dir = ensure_dir(Path(args.output_dir) / "figures")
    _plot_curve(curves, "k", fig_dir / "sensitivity_k_curve.pdf")
    _plot_curve(curves, "tau", fig_dir / "sensitivity_tau_curve.pdf")
    _plot_curve(curves, "holdout_ratio", fig_dir / "sensitivity_holdout_ratio_curve.pdf")
    _plot_curve(curves, "latent_dim", fig_dir / "sensitivity_latent_dim_curve.pdf")
    _plot_curve(curves[curves["metric"].eq("hidden_edge_precision_at_k")], "k", fig_dir / "sensitivity_attachment_precision_curve.pdf")
    _plot_curve(curves[curves["metric"].eq("feature_mmd")], "k", fig_dir / "sensitivity_feature_mmd_curve.pdf")
    _plot_curve(curves[curves["metric"].isin(["attachment_auc", "attachment_ap"])], "k", fig_dir / "sensitivity_auc_ap_curve.pdf")
    ensure_dir(fig_dir / "sensitivity_heatmaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
