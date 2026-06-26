from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import torch

from agn_real.data import load_real_dataset
from agn_real.experiments.run_real_node_holdout import run_rc_agn
from agn_real.splits import stratified_node_holdout
from agn_real.utils import set_seed


@dataclass
class ExampleSelection:
    hidden_old: List[int]
    observed_old: List[int]
    generated: List[int]
    generated_edges_old: Dict[int, List[int]]


def _undirected_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if edge_index.numel() == 0:
        return edges
    for s, d in edge_index.t().tolist():
        if int(s) == int(d):
            continue
        a, b = sorted((int(s), int(d)))
        edges.add((a, b))
    return edges


def _neighbors(edges: Iterable[tuple[int, int]], nodes: Sequence[int]) -> Dict[int, set[int]]:
    node_set = set(int(n) for n in nodes)
    out = {int(n): set() for n in nodes}
    for a, b in edges:
        if a in node_set:
            out[a].add(b)
        if b in node_set:
            out[b].add(a)
    return out


def _candidate_hidden_pairs(split, full_edges: set[tuple[int, int]]) -> list[tuple[int, int, set[int]]]:
    hidden = [int(v) for v in split.hidden_node_ids.tolist()]
    observed_lookup = set(split.old_to_observed.keys())
    neigh = _neighbors(full_edges, hidden)
    candidates: list[tuple[int, int, set[int]]] = []
    for i, h1 in enumerate(hidden):
        obs1 = {v for v in neigh[h1] if v in observed_lookup}
        if not 2 <= len(obs1) <= 8:
            continue
        for h2 in hidden[i + 1 :]:
            obs2 = {v for v in neigh[h2] if v in observed_lookup}
            union = obs1 | obs2
            if 5 <= len(union) <= 13:
                candidates.append((h1, h2, union))
    candidates.sort(key=lambda item: (len(item[2]), item[0], item[1]))
    return candidates


def _generated_attachment_map(split, result, max_edges_per_generated: int = 4) -> Dict[int, List[int]]:
    n_obs = split.observed_data.num_nodes
    edge_scores = result.get("edge_scores")
    edge_index = result["edge_index"]
    by_gen: Dict[int, list[tuple[int, float]]] = {}
    for idx, (s, d) in enumerate(edge_index.t().tolist()):
        s, d = int(s), int(d)
        if s < n_obs or d >= n_obs:
            continue
        gen = s - n_obs
        old_target = int(split.observed_to_old[d])
        score = float(edge_scores[idx]) if edge_scores is not None and len(edge_scores) > idx else 1.0
        by_gen.setdefault(gen, []).append((old_target, score))
    out: Dict[int, List[int]] = {}
    for gen, values in by_gen.items():
        unique: Dict[int, float] = {}
        for old, score in values:
            unique[old] = max(unique.get(old, float("-inf")), score)
        ordered = sorted(unique.items(), key=lambda item: (-item[1], item[0]))
        out[gen] = [old for old, _ in ordered[:max_edges_per_generated]]
    return out


def select_example(split, result) -> ExampleSelection | None:
    full_edges = _undirected_edges(split.full_data_reference.edge_index)
    gen_edges = _generated_attachment_map(split, result)
    hidden = [int(v) for v in split.hidden_node_ids.tolist()]
    observed_lookup = set(split.old_to_observed.keys())
    neigh = _neighbors(full_edges, hidden)

    single_candidates: list[tuple[int, int, int, set[int], list[tuple[int, int, List[int]]]]] = []
    for h in hidden:
        observed = {v for v in neigh[h] if v in observed_lookup}
        if not 3 <= len(observed) <= 10:
            continue
        observed_edges = sum(1 for a, b in full_edges if a in observed and b in observed)
        if observed_edges < 2:
            continue
        scored: list[tuple[int, int, List[int]]] = []
        for gen, targets in gen_edges.items():
            local = [t for t in targets if t in observed]
            if local:
                scored.append((len(local), gen, local))
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            single_candidates.append((observed_edges, len(observed), h, observed, scored))
    single_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if single_candidates:
        _, _, h, observed, scored = single_candidates[0]
        gen = scored[0][1]
        return ExampleSelection(
            hidden_old=[h],
            observed_old=sorted(observed),
            generated=[gen],
            generated_edges_old={gen: scored[0][2][:3]},
        )

    for h1, h2, observed in _candidate_hidden_pairs(split, full_edges):
        scored: list[tuple[int, int, List[int]]] = []
        for gen, targets in gen_edges.items():
            local = [t for t in targets if t in observed]
            if local:
                scored.append((len(local), gen, local))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if len(scored) < 2:
            continue
        selected_gen = [scored[0][1], scored[1][1]]
        shown_edges = {gen: scored[idx][2][:3] for idx, gen in enumerate(selected_gen)}
        final_observed = set(observed)
        for targets in shown_edges.values():
            final_observed.update(targets)
        if 5 <= len(final_observed) <= 14:
            return ExampleSelection(
                hidden_old=[h1, h2],
                observed_old=sorted(final_observed),
                generated=selected_gen,
                generated_edges_old=shown_edges,
            )
    return None


def _normalize_positions(pos: Dict[object, tuple[float, float]], x0: float, y0: float, w: float, h: float) -> Dict[object, tuple[float, float]]:
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    padx, pady = 0.12 * w, 0.18 * h
    out = {}
    for node, (x, y) in pos.items():
        xx = x0 + padx + (x - minx) / max(maxx - minx, 1e-9) * (w - 2 * padx)
        yy = y0 + pady + (y - miny) / max(maxy - miny, 1e-9) * (h - 2 * pady)
        out[node] = (xx, yy)
    return out


def build_figure_spec(selection: ExampleSelection, split) -> dict:
    full_edges = _undirected_edges(split.full_data_reference.edge_index)
    hidden = set(selection.hidden_old)
    observed = set(selection.observed_old)
    local_nodes = sorted(hidden | observed)
    local_edges = [e for e in full_edges if e[0] in local_nodes and e[1] in local_nodes]

    # Layout is fixed from the full reference subgraph and reused in all panels.
    g_layout = nx.Graph()
    g_layout.add_nodes_from(local_nodes)
    g_layout.add_edges_from(local_edges)
    raw_pos = nx.spring_layout(g_layout, seed=7, k=0.9, iterations=200)
    W, H = 1080, 430
    panels = [
        ("A", "Full reference subgraph", 28, 78),
        ("B", "Observed incomplete subgraph", 375, 78),
        ("C", "AGN-augmented subgraph", 722, 78),
    ]
    panel_w, panel_h = 320, 285
    panel_positions = {
        x0: _normalize_positions(raw_pos, x0, y0, panel_w, panel_h)
        for _, _, x0, y0 in panels
    }
    if len(selection.generated) == 1:
        gen_anchor = {selection.generated[0]: (0.74, 0.20)}
    else:
        gen_anchor = {
            selection.generated[0]: (0.12, 0.30),
            selection.generated[1]: (0.88, 0.38),
        }

    spec = {
        "width": W,
        "height": H,
        "panels": [{"label": label, "title": title, "x": x0, "y": y0, "w": panel_w, "h": panel_h} for label, title, x0, y0 in panels],
        "nodes": [],
        "edges": [],
        "metadata": {
            "dataset": "Cora",
            "seed": int(split.seed),
            "holdout_ratio": float(split.holdout_ratio),
            "hidden_old": selection.hidden_old,
            "generated": selection.generated,
        },
    }
    for idx, (_, _, x0, y0) in enumerate(panels):
        pos = panel_positions[x0]
        panel_label = panels[idx][0]
        if idx == 0:
            for a, b in local_edges:
                spec["edges"].append({"panel": panel_label, "source": pos[a], "target": pos[b], "kind": "observed"})
            for n in observed:
                spec["nodes"].append({"panel": panel_label, "id": int(n), "pos": pos[n], "kind": "observed"})
            for n in hidden:
                spec["nodes"].append({"panel": panel_label, "id": int(n), "pos": pos[n], "kind": "hidden"})
        elif idx == 1:
            for a, b in local_edges:
                if a in observed and b in observed:
                    spec["edges"].append({"panel": panel_label, "source": pos[a], "target": pos[b], "kind": "observed"})
                elif (a in hidden or b in hidden) and (a in observed or b in observed):
                    ha = a if a in hidden else b
                    oa = b if a in hidden else a
                    spec["edges"].append({"panel": panel_label, "source": pos[ha], "target": pos[oa], "kind": "removed"})
            for n in observed:
                spec["nodes"].append({"panel": panel_label, "id": int(n), "pos": pos[n], "kind": "observed"})
            for n in hidden:
                spec["nodes"].append({"panel": panel_label, "id": int(n), "pos": pos[n], "kind": "hidden_ghost"})
        else:
            for a, b in local_edges:
                if a in observed and b in observed:
                    spec["edges"].append({"panel": panel_label, "source": pos[a], "target": pos[b], "kind": "observed"})
            for n in observed:
                spec["nodes"].append({"panel": panel_label, "id": int(n), "pos": pos[n], "kind": "observed"})
            gen_pos: Dict[int, tuple[float, float]] = {}
            for gen, (rx, ry) in gen_anchor.items():
                gx = x0 + rx * panel_w
                gy = y0 + ry * panel_h
                gen_pos[gen] = (gx, gy)
                spec["nodes"].append({"panel": panel_label, "id": int(gen), "pos": (gx, gy), "kind": "generated"})
            for gen, targets in selection.generated_edges_old.items():
                for old in targets:
                    if old in pos:
                        spec["edges"].append({"panel": panel_label, "source": gen_pos[gen], "target": pos[old], "kind": "agn"})
    return spec


def write_svg(spec: dict, path: Path) -> None:
    def line(e: dict) -> str:
        color = {"observed": "#222222", "removed": "#b8b8b8", "agn": "#ff5a00"}[e["kind"]]
        width = {"observed": "1.2", "removed": "1.0", "agn": "1.8"}[e["kind"]]
        dash = ' stroke-dasharray="5 4"' if e["kind"] == "removed" else ""
        x1, y1 = e["source"]
        x2, y2 = e["target"]
        return f'<line x1="{x1:.2f}" y1="{spec["height"]-y1:.2f}" x2="{x2:.2f}" y2="{spec["height"]-y2:.2f}" stroke="{color}" stroke-width="{width}"{dash}/>'

    def node(n: dict) -> str:
        x, y = n["pos"]
        yy = spec["height"] - y
        kind = n["kind"]
        if kind == "observed":
            return f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="7.5" fill="#2b7ed8" stroke="#111111" stroke-width="1.2"/>'
        if kind == "hidden":
            return f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="8" fill="#d2d2d2" stroke="#666666" stroke-width="1.4"/>'
        if kind == "hidden_ghost":
            return f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="8" fill="none" stroke="#888888" stroke-width="1.2" stroke-dasharray="4 3"/>'
        return f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="8.2" fill="#ffa735" stroke="#111111" stroke-width="1.2"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{spec["width"]}" height="{spec["height"]}" viewBox="0 0 {spec["width"]} {spec["height"]}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for p in spec["panels"]:
        parts.append(f'<rect x="{p["x"]}" y="{spec["height"]-p["y"]-p["h"]}" width="{p["w"]}" height="{p["h"]}" rx="8" fill="#fbfcfd" stroke="#cfd6dc"/>')
        parts.append(f'<text x="{p["x"]+12}" y="{spec["height"]-p["y"]-p["h"]+28}" font-family="Arial" font-size="24" font-weight="700">{p["label"]}</text>')
        parts.append(f'<text x="{p["x"]+p["w"]/2}" y="{spec["height"]-p["y"]-p["h"]+25}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">{p["title"]}</text>')
    parts.extend(line(e) for e in spec["edges"])
    parts.extend(node(n) for n in spec["nodes"])
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def build(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a real AGN graph-augmentation example figure.")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout_ratio", type=float, default=0.05)
    parser.add_argument("--output", default="AGN-ELSEVIER-REVISION/figures/real_agn_graph_augmentation_example.pdf")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--attachment_epochs", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    data, _ = load_real_dataset(args.dataset, args.data_root)
    split = stratified_node_holdout(data, args.holdout_ratio, args.seed, "degree_stratified")
    run_args = SimpleNamespace(
        hidden_dim=64,
        latent_dim=32,
        encoder="gcn",
        epochs=args.epochs,
        attachment_epochs=args.attachment_epochs,
        k=args.k,
        attachment_threshold=0.5,
        agn_feature_mode="role_interpolation",
        agn_attachment_mode="hybrid",
        feature_mmd_weight=0.5,
        feature_centroid_weight=0.5,
        interpolation_lambda=0.5,
        hybrid_alpha=0.7,
        hybrid_beta=0.2,
        hybrid_gamma=0.0,
        hybrid_delta=0.1,
        device="cpu",
    )
    result = run_rc_agn(split.observed_data, len(split.hidden_node_ids), args.seed, run_args)
    selection = select_example(split, result)
    if selection is None:
        raise RuntimeError("No compact, interpretable held-out subgraph with local AGN attachments was found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = build_figure_spec(selection, split)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    write_svg(spec, output.with_suffix(".svg"))
    print(
        f"wrote {json_path} and {output.with_suffix('.svg')} using {args.dataset}, seed={args.seed}, "
        f"holdout_ratio={args.holdout_ratio}, hidden_old={selection.hidden_old}, "
        f"generated={selection.generated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
