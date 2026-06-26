#!/usr/bin/env python3
"""
Build publication figures (PDF + PNG) in the project ROOT.
Requires results/upgraded/all_results.json and results/upgraded/viz_cache/*.pkl from experiments.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agn_general.config import BASE_DIR, PAPER_FIGURES_DIR, PAPER_TRIPTYCH_DATASETS
from agn_general.paper_figure_export import (
    figure_architecture_schematic,
    figure_degree_triptych,
    figure_edge_composition_faceted,
    figure_metrics_triptych,
    figure_network_triptych,
    figure_novelty_triptych,
    figure_pca_triptych,
)


def load_results(path: Path):
    with open(path) as f:
        return json.load(f)


def main():
    root = Path(BASE_DIR)
    out = Path(PAPER_FIGURES_DIR)
    res_path = root / "results" / "upgraded" / "all_results.json"
    viz_dir = root / "results" / "upgraded" / "viz_cache"

    figure_architecture_schematic(str(out / "fig2_architecture.pdf"), str(out / "fig2_architecture.png"))

    if not res_path.exists():
        print(f"Missing {res_path}; run run_experiments_upgraded.py first.")
        return

    results = load_results(res_path)
    labels = ["Community-SBM", "Multi-comm.", "Scale-free sparse"]
    trip_names = PAPER_TRIPTYCH_DATASETS

    # --- Fig 3–8 from viz pickles (no_gg / AGN) ---
    net_panels = []
    met_panels = []
    deg_panels = []
    nov_panels = []
    pca_panels = []

    for ds, lab in zip(trip_names, labels):
        pkl = viz_dir / f"{ds}_no_gg.pkl"
        if not pkl.exists():
            print(f"Warning: missing viz cache {pkl}, skip dataset {ds} in triptychs")
            continue
        with open(pkl, "rb") as f:
            blob = pickle.load(f)
        G, G_aug = blob["G"], blob["G_aug"]
        feat, gen = blob["features"], blob["generated_features"]
        from agn_general.evaluation import compute_topology_metrics

        net_panels.append((G, G_aug, lab))
        met_panels.append((compute_topology_metrics(G), compute_topology_metrics(G_aug), lab))
        deg_panels.append((G, G_aug, lab))
        nov_panels.append((feat, gen, lab))
        pca_panels.append((feat, gen, lab))

    if len(net_panels) == len(trip_names):
        figure_network_triptych(net_panels, str(out / "fig3_network_comparison.pdf"), str(out / "fig3_network_comparison.png"))
        figure_metrics_triptych(met_panels, str(out / "fig4_normalized_metrics.pdf"), str(out / "fig4_normalized_metrics.png"))
        figure_degree_triptych(deg_panels, str(out / "fig5_degree_distribution.pdf"), str(out / "fig5_degree_distribution.png"))
        figure_novelty_triptych(nov_panels, str(out / "fig7_novelty_histograms.pdf"), str(out / "fig7_novelty_histograms.png"))
        figure_pca_triptych(pca_panels, str(out / "fig8_pca.pdf"), str(out / "fig8_pca.png"))

    ec_panels = []
    for ds, lab in zip(trip_names, labels):
        row_o = next(
            (r for r in results if r.get("success") and r.get("dataset") == ds and r.get("variant") == "original"),
            None,
        )
        row_n = next(
            (r for r in results if r.get("success") and r.get("dataset") == ds and r.get("variant") == "no_gg"),
            None,
        )
        if not row_o or not row_n:
            continue
        ec_o = row_o["diagnostics"]["edge_composition"]
        ec_n = row_n["diagnostics"]["edge_composition"]
        t_o = (ec_o.get("original_original", 0), ec_o.get("generated_original", 0), ec_o.get("generated_generated", 0))
        t_n = (ec_n.get("original_original", 0), ec_n.get("generated_original", 0), ec_n.get("generated_generated", 0))
        ec_panels.append((t_o, t_n, lab))

    if len(ec_panels) == len(trip_names):
        figure_edge_composition_faceted(ec_panels, str(out / "fig6_edge_composition.pdf"), str(out / "fig6_edge_composition.png"))

    print(f"Paper figures written to {out}")


if __name__ == "__main__":
    main()
