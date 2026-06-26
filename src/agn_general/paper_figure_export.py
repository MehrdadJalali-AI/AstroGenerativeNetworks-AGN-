"""
Assemble multi-panel paper figures (IEEE) from graph / feature arrays.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import compute_topology_metrics, novelty_analysis
from .paper_style import apply_paper_matplotlib_style


def _subsample_augmented(G_before: nx.Graph, G_after: nx.Graph, max_viz_nodes: int = 500):
    n_orig = G_before.number_of_nodes()
    if G_before.number_of_nodes() <= max_viz_nodes:
        return G_before.copy(), G_after.copy()
    sample_before = np.random.choice(list(G_before.nodes()), size=max_viz_nodes, replace=False)
    G_before_viz = G_before.subgraph(sample_before).copy()
    generated_nodes = [n for n in G_after.nodes() if n >= n_orig]
    n_orig_sample = max(max_viz_nodes - len(generated_nodes), 1)
    sampled_orig = np.random.choice([n for n in G_after.nodes() if n < n_orig], size=min(n_orig_sample, n_orig), replace=False)
    sample_after = list(sampled_orig) + generated_nodes
    G_after_viz = G_after.subgraph(sample_after).copy()
    return G_before_viz, G_after_viz


def draw_before_after_axes(
    G_before: nx.Graph,
    G_after: nx.Graph,
    n_orig: int,
    ax_before,
    ax_after,
    seed: int = 42,
    max_viz_nodes: int = 500,
):
    Gbv, Gav = _subsample_augmented(G_before, G_after, max_viz_nodes)
    pos_b = nx.spring_layout(Gbv, seed=seed, k=0.5, iterations=50)
    nx.draw_networkx_nodes(Gbv, pos_b, ax=ax_before, node_size=18, node_color="#1864AB", alpha=0.75, linewidths=0)
    nx.draw_networkx_edges(Gbv, pos_b, ax=ax_before, alpha=0.12, width=0.35)
    ax_before.set_axis_off()

    pos_a = nx.spring_layout(Gav, seed=seed, k=0.5, iterations=50)
    orig_v = [n for n in Gav.nodes() if n < n_orig]
    gen_v = [n for n in Gav.nodes() if n >= n_orig]
    nx.draw_networkx_nodes(Gav, pos_a, nodelist=orig_v, ax=ax_after, node_size=18, node_color="#1864AB", alpha=0.75, linewidths=0)
    nx.draw_networkx_nodes(Gav, pos_a, nodelist=gen_v, ax=ax_after, node_size=36, node_color="#E03131", alpha=0.85, linewidths=0)
    nx.draw_networkx_edges(Gav, pos_a, ax=ax_after, alpha=0.12, width=0.35)
    ax_after.set_axis_off()


def figure_network_triptych(
    panels: List[Tuple[nx.Graph, nx.Graph, str]],
    out_pdf: str,
    out_png: Optional[str] = None,
):
    """panels: list of (G_before, G_after, column_title)."""
    apply_paper_matplotlib_style()
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, (Gb, Ga, title) in enumerate(panels):
        n_orig = Gb.number_of_nodes()
        draw_before_after_axes(Gb, Ga, n_orig, axes[0, j], axes[1, j])
        axes[0, j].set_title(f"{title}\n(before)", fontweight="600", fontsize=10)
        axes[1, j].set_title("after AGN", fontweight="600", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)


def figure_metrics_triptych(
    panels: List[Tuple[dict, dict, str]],
    out_pdf: str,
    out_png: Optional[str] = None,
):
    apply_paper_matplotlib_style()
    key_metrics = ["density", "avg_degree", "avg_clustering", "modularity", "avg_shortest_path_length", "assortativity"]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, (before, after, title) in zip(axes, panels):
        avail = [m for m in key_metrics if m in before and np.isfinite(before.get(m, np.nan))]
        if not avail:
            ax.set_axis_off()
            continue
        x = np.arange(len(avail))
        w = 0.36
        max_vals = [max(abs(before[m]), abs(after[m])) for m in avail]
        bnorm = [before[m] / (mv + 1e-10) for m, mv in zip(avail, max_vals)]
        anorm = [after[m] / (mv + 1e-10) for m, mv in zip(avail, max_vals)]
        ax.bar(x - w / 2, bnorm, w, label="before", color="#1864AB", alpha=0.88)
        ax.bar(x + w / 2, anorm, w, label="after", color="#E03131", alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in avail], rotation=0, fontsize=8)
        ax.set_ylabel("normalized")
        ax.set_title(title, fontweight="600")
        ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)


def figure_degree_triptych(
    panels: List[Tuple[nx.Graph, nx.Graph, str]],
    out_pdf: str,
    out_png: Optional[str] = None,
):
    apply_paper_matplotlib_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (Gb, Ga, title) in zip(axes, panels):
        db = [d for _, d in Gb.degree()]
        da = [d for _, d in Ga.degree()]
        ax.hist(db, bins=40, alpha=0.65, density=True, color="#1864AB", label="before", edgecolor="none")
        ax.hist(da, bins=40, alpha=0.55, density=True, color="#E03131", label="after", edgecolor="none")
        ax.set_xlabel("degree")
        ax.set_ylabel("density")
        ax.set_title(title, fontweight="600")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)


def figure_edge_composition_faceted(
    panels: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]],
    out_pdf: str,
    out_png: Optional[str] = None,
):
    """
    panels: list of ( (oo,go,gg)_AGN_original, (oo,go,gg)_AGN, column_title ).
    Two stacked bars per panel: diagnostic baseline vs recommended AGN.
    """
    apply_paper_matplotlib_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.8))
    if n == 1:
        axes = [axes]
    colors = ["#495057", "#0B7285", "#E8590C"]
    for ax, (orig_v, agn_v, title) in zip(axes, panels):
        xs = [0, 1]
        for xi, (oo, go, gg) in zip(xs, [orig_v, agn_v]):
            ax.bar([xi], [oo], color=colors[0], width=0.55, label="orig–orig" if xi == 0 else None)
            ax.bar([xi], [go], bottom=[oo], color=colors[1], width=0.55, label="gen–orig" if xi == 0 else None)
            ax.bar([xi], [gg], bottom=[oo + go], color=colors[2], width=0.55, label="gen–gen" if xi == 0 else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["AGN-orig.", "AGN"], fontsize=9)
        ax.set_ylabel("new edges")
        ax.set_title(title, fontweight="600")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:3], labels[:3], loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)


def figure_novelty_triptych(
    panels: List[Tuple[np.ndarray, np.ndarray, str]],
    out_pdf: str,
    out_png: Optional[str] = None,
):
    apply_paper_matplotlib_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (orig_f, gen_f, title) in zip(axes, panels):
        nov = novelty_analysis(orig_f, gen_f)
        md = nov.get("min_distance_to_original", [])
        if len(md) > 0:
            ax.hist(md, bins=28, color="#2B8A3E", alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.axvline(float(np.mean(md)), color="#E03131", linestyle="--", linewidth=1.2, label="mean")
        ax.set_xlabel(r"min$_i$ $(1 - \cos)$ to originals")
        ax.set_ylabel("count")
        ax.set_title(title, fontweight="600")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)


def figure_pca_triptych(
    panels: List[Tuple[np.ndarray, np.ndarray, str]],
    out_pdf: str,
    out_png: Optional[str] = None,
):
    apply_paper_matplotlib_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (orig_f, gen_f, title) in zip(axes, panels):
        all_x = np.vstack([orig_f, gen_f])
        pca = PCA(n_components=2)
        z = pca.fit_transform(all_x)
        n0 = len(orig_f)
        ax.scatter(z[:n0, 0], z[:n0, 1], s=14, c="#1864AB", alpha=0.55, label="original", linewidths=0)
        ax.scatter(z[n0:, 0], z[n0:, 1], s=28, c="#E03131", alpha=0.75, marker="^", label="generated", linewidths=0)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(title, fontweight="600")
        ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)


def figure_architecture_schematic(out_pdf: str, out_png: Optional[str] = None):
    """Fig. 2: encoder → latent → node decoder (inference) + training-only inner-product edge scorer."""
    apply_paper_matplotlib_style()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F1F3F5"):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0, edgecolor="#212529", facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="600")

    box(0.3, 2.2, 1.8, 1.1, "GCN\nencoder")
    box(2.6, 2.2, 1.5, 1.1, "Latent\n$z \\sim q(z|X,A)$")
    box(4.5, 2.2, 1.8, 1.1, "Node decoder\n(MLP)")
    box(6.9, 2.2, 1.6, 1.1, "Similarity\nattachment")

    box(2.4, 0.35, 2.4, 0.95, "Inner-product edge\nscores (train only)", fc="#E7F5FF")

    for x1, x2 in [(2.1, 2.6), (4.1, 4.5), (6.3, 6.9)]:
        ax.add_patch(FancyArrowPatch((x1, 2.75), (x2, 2.75), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#343A40"))
    ax.add_patch(FancyArrowPatch((8.5, 2.75), (9.4, 2.75), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#343A40"))
    ax.text(9.45, 2.75, "$G'$", va="center", fontsize=11, fontweight="600")

    ax.add_patch(
        FancyArrowPatch(
            (3.4, 2.2), (3.4, 1.35),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#495057", linestyle="--",
        )
    )
    ax.text(3.55, 1.7, "train", fontsize=8, color="#495057", rotation=90, va="center")

    ax.text(5.0, 3.95, "AGN pipeline (inference: solid path; edge inner product: dashed, training only)", ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)
