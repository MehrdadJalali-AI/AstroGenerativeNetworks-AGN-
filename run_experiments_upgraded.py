#!/usr/bin/env python3
"""
Full AGN experiment grid: L = L_recon + gamma L_feat + beta L_KL, validation early stopping,
synthetic + real datasets, diagnostics, and viz cache for paper figures.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agn_general.config import (
    DEVICE,
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_GCN_LAYERS,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    RESULTS_DIR,
    BETA,
    GAMMA,
    EARLY_STOPPING_PATIENCE,
    USE_VALIDATION_EARLY_STOPPING,
    EXPERIMENT_DATASETS,
    REAL_EXPERIMENT_DATASETS,
    PAPER_TRIPTYCH_DATASETS,
    NUM_GENERATED_NODES,
    NUM_GENERATED_NODES_SMALL,
    INSERTION_VARIANTS,
)
from agn_general.data_loader import load_dataset
from agn_general.datasets_extended import load_dataset_extended
from agn_general.model import GraphEncoder, NodeDecoder, VGAE
from agn_general.training import run_training
from agn_general.generation_variants import generate_and_insert_variant
from agn_general.diagnostics import compute_insertion_diagnostics, save_diagnostics
from agn_general.novelty_evaluation import compute_enhanced_novelty, save_novelty_metrics
from agn_general.evaluation import compute_topology_metrics
from agn_general.comprehensive_evaluation import run_comprehensive_evaluation
from agn_general.ablation_analysis import run_ablation_study

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    return obj


def ablation_json_summary(abl: dict) -> dict:
    out = {"dataset": abl.get("dataset"), "ablations": {}, "sensitivity": abl.get("sensitivity")}
    for name, block in abl.get("ablations", {}).items():
        if block is None:
            out["ablations"][name] = None
            continue
        topo = block.get("topology") or {}
        out["ablations"][name] = {
            "method": block.get("method"),
            "topology": {k: float(topo[k]) for k in topo if isinstance(topo[k], (int, float, np.floating))},
        }
    return out


def _load_any_dataset(dataset_name: str, portion: float = 1.0):
    try:
        return load_dataset_extended(dataset_name, portion=portion)
    except Exception:
        return load_dataset(dataset_name, portion=portion)


def _num_generated_for(dataset_name: str, base_m: int) -> int:
    if dataset_name in REAL_EXPERIMENT_DATASETS:
        return NUM_GENERATED_NODES_SMALL
    return base_m


def run_single_experiment(
    dataset_name: str,
    variant: str,
    num_generated: int,
    k: int = 10,
    tau: float = 0.5,
    hidden_dim: int = HIDDEN_DIM,
    latent_dim: int = LATENT_DIM,
    seed: int = 42,
    epochs: int = EPOCHS,
    viz_dir: Path = None,
):
    print(f"\n{'='*80}")
    print(f"Experiment: {dataset_name} | {variant} | M={num_generated} | k={k} | tau={tau}")
    print(f"{'='*80}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    results = {
        "dataset": dataset_name,
        "variant": variant,
        "num_generated": num_generated,
        "k": k,
        "tau": tau,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "seed": seed,
        "beta": BETA,
        "gamma": GAMMA,
    }

    try:
        G, features, edge_index, scaler, feat_min, feat_max = _load_any_dataset(dataset_name, portion=1.0)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        edge_index_tensor = edge_index.to(DEVICE)

        input_dim = features.shape[1]
        encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, NUM_GCN_LAYERS)
        decoder = NodeDecoder(latent_dim, hidden_dim, input_dim)
        model = VGAE(encoder, decoder).to(DEVICE)

        ckpt_name = f"agn_{dataset_name}_{variant}_{seed}.pth"
        train_out = run_training(
            model,
            features_tensor,
            edge_index_tensor,
            epochs=epochs,
            lr=LEARNING_RATE,
            beta=BETA,
            gamma=GAMMA,
            patience=EARLY_STOPPING_PATIENCE,
            use_validation_early_stopping=USE_VALIDATION_EARLY_STOPPING,
            checkpoint_name=ckpt_name,
        )
        results["training"] = {
            "best_epoch": train_out["best_epoch"],
            "best_val_or_train_loss": train_out["best_val_or_train_loss"],
            "test_loss": train_out["test_loss"],
            "test_parts": train_out["test_parts"],
            "train_losses_tail": train_out["train_losses"][-10:],
            "val_losses_tail": [x for x in train_out["val_losses"][-10:] if x == x],
        }

        G_augmented, generated_features, generated_node_ids = generate_and_insert_variant(
            model, G, features, variant=variant, num_samples=num_generated, k_neighbors=k, threshold=tau
        )

        diagnostics = compute_insertion_diagnostics(
            G, G_augmented, features, generated_features, generated_node_ids
        )
        results["diagnostics"] = diagnostics

        novelty_metrics = compute_enhanced_novelty(features, generated_features)
        results["novelty"] = novelty_metrics

        orig_metrics = compute_topology_metrics(G)
        aug_metrics = compute_topology_metrics(G_augmented)
        results["topology"] = {
            "original": orig_metrics,
            "augmented": aug_metrics,
            "changes": {
                "density": aug_metrics["density"] - orig_metrics["density"],
                "clustering": aug_metrics["avg_clustering"] - orig_metrics["avg_clustering"],
                "modularity": aug_metrics.get("modularity", 0) - orig_metrics.get("modularity", 0),
                "path_length": aug_metrics.get("avg_shortest_path_length", 0)
                - orig_metrics.get("avg_shortest_path_length", 0),
                "assortativity": aug_metrics.get("assortativity", 0) - orig_metrics.get("assortativity", 0),
            },
        }

        results["success"] = True

        if viz_dir is not None and variant == "no_gg" and dataset_name in PAPER_TRIPTYCH_DATASETS:
            viz_dir.mkdir(parents=True, exist_ok=True)
            with open(viz_dir / f"{dataset_name}_no_gg.pkl", "wb") as vf:
                pickle.dump(
                    {
                        "G": G,
                        "G_aug": G_augmented,
                        "features": features,
                        "generated_features": generated_features,
                    },
                    vf,
                )

        if results.get("success"):
            save_diagnostics(diagnostics, dataset_name, variant, str(Path(RESULTS_DIR) / "upgraded" / "diagnostics"))
            save_novelty_metrics(
                novelty_metrics, dataset_name, variant, str(Path(RESULTS_DIR) / "upgraded")
            )

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        results["success"] = False
        results["error"] = str(e)

    return results


def run_ablation_study_karate(base_config):
    ablation_results = []
    for M in [25, 50, 100, 200]:
        r = run_single_experiment(
            "karate",
            "no_gg",
            num_generated=M,
            k=base_config["k"],
            tau=base_config["tau"],
            hidden_dim=base_config["hidden_dim"],
            latent_dim=base_config["latent_dim"],
            seed=base_config["seed"],
        )
        r["ablation_type"] = "M"
        r["ablation_value"] = M
        ablation_results.append(r)
    for k in [5, 10, 15, 20]:
        r = run_single_experiment(
            "karate",
            "no_gg",
            num_generated=base_config["num_generated"],
            k=k,
            tau=base_config["tau"],
            hidden_dim=base_config["hidden_dim"],
            latent_dim=base_config["latent_dim"],
            seed=base_config["seed"],
        )
        r["ablation_type"] = "k"
        r["ablation_value"] = k
        ablation_results.append(r)
    for tau in [0.3, 0.5, 0.7, 0.9]:
        r = run_single_experiment(
            "karate",
            "no_gg",
            num_generated=base_config["num_generated"],
            k=base_config["k"],
            tau=tau,
            hidden_dim=base_config["hidden_dim"],
            latent_dim=base_config["latent_dim"],
            seed=base_config["seed"],
        )
        r["ablation_type"] = "tau"
        r["ablation_value"] = tau
        ablation_results.append(r)
    for hidden_dim in [16, 32, 64, 128]:
        r = run_single_experiment(
            "karate",
            "no_gg",
            num_generated=base_config["num_generated"],
            k=base_config["k"],
            tau=base_config["tau"],
            hidden_dim=hidden_dim,
            latent_dim=base_config["latent_dim"],
            seed=base_config["seed"],
        )
        r["ablation_type"] = "hidden_dim"
        r["ablation_value"] = hidden_dim
        ablation_results.append(r)
    for latent_dim in [8, 16, 32]:
        r = run_single_experiment(
            "karate",
            "no_gg",
            num_generated=base_config["num_generated"],
            k=base_config["k"],
            tau=base_config["tau"],
            hidden_dim=base_config["hidden_dim"],
            latent_dim=latent_dim,
            seed=base_config["seed"],
        )
        r["ablation_type"] = "latent_dim"
        r["ablation_value"] = latent_dim
        ablation_results.append(r)
    return ablation_results


def run_posthoc_baselines_and_ablation():
    """Train once on Community-SBM (`karate` synthetic) and run baseline / ablation bundles."""
    upgraded = Path(RESULTS_DIR) / "upgraded"
    G, features, edge_index, _, _, _ = _load_any_dataset("karate", portion=1.0)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    edge_index_tensor = edge_index.to(DEVICE)
    input_dim = features.shape[1]
    encoder = GraphEncoder(input_dim, HIDDEN_DIM, LATENT_DIM, NUM_GCN_LAYERS)
    decoder = NodeDecoder(LATENT_DIM, HIDDEN_DIM, input_dim)
    model = VGAE(encoder, decoder).to(DEVICE)
    run_training(
        model,
        features_tensor,
        edge_index_tensor,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        beta=BETA,
        gamma=GAMMA,
        patience=EARLY_STOPPING_PATIENCE,
        use_validation_early_stopping=USE_VALIDATION_EARLY_STOPPING,
        checkpoint_name="posthoc_karate_baselines.pth",
    )
    print("Running comprehensive baseline evaluation (Community-SBM / karate synthetic)...")
    comp = run_comprehensive_evaluation(
        model,
        G,
        features,
        "karate",
        num_generated=NUM_GENERATED_NODES,
        k=10,
        threshold=0.5,
        seed=RANDOM_SEED,
    )
    with open(upgraded / "baselines_karate.json", "w") as f:
        json.dump(json_safe(comp), f, indent=2)

    print("Running ablation study (same trained model)...")
    abl = run_ablation_study(model, G, features, "karate", num_generated=NUM_GENERATED_NODES)
    with open(upgraded / "ablation_karate.json", "w") as f:
        json.dump(json_safe(ablation_json_summary(abl)), f, indent=2)


def main():
    import sys

    fast_paper = "--fast-paper" in sys.argv

    output_base = Path(RESULTS_DIR) / "upgraded"
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "diagnostics").mkdir(exist_ok=True)
    (output_base / "raw" / "novelty").mkdir(parents=True, exist_ok=True)
    (output_base / "tables").mkdir(exist_ok=True)
    viz_dir = output_base / "viz_cache"
    viz_dir.mkdir(exist_ok=True)

    if fast_paper:
        # Triptych regimes (full variant grid) + small real graphs (AGN / no_gg only for appendix)
        datasets = list(PAPER_TRIPTYCH_DATASETS) + list(REAL_EXPERIMENT_DATASETS)
        variants = list(INSERTION_VARIANTS)
    else:
        datasets = list(EXPERIMENT_DATASETS) + list(REAL_EXPERIMENT_DATASETS)
        variants = list(INSERTION_VARIANTS)

    base_config = {
        "num_generated": NUM_GENERATED_NODES,
        "k": 10,
        "tau": 0.5,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "seed": RANDOM_SEED,
    }

    all_results = []

    print("\n" + "=" * 80)
    print("MAIN EXPERIMENTS")
    print("=" * 80)
    for dataset_name in datasets:
        mgen = _num_generated_for(dataset_name, base_config["num_generated"])
        variant_list = variants
        if fast_paper and dataset_name in REAL_EXPERIMENT_DATASETS:
            variant_list = ["no_gg"]
        for variant in variant_list:
            try:
                result = run_single_experiment(
                    dataset_name,
                    variant,
                    num_generated=mgen,
                    k=base_config["k"],
                    tau=base_config["tau"],
                    hidden_dim=base_config["hidden_dim"],
                    latent_dim=base_config["latent_dim"],
                    seed=base_config["seed"],
                    viz_dir=viz_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Failed: {dataset_name} | {variant}: {e}")

    if not fast_paper:
        print("\n" + "=" * 80)
        print("ABLATION STUDY (karate synthetic)")
        print("=" * 80)
        all_results.extend(run_ablation_study_karate(base_config))

    results_file = output_base / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(json_safe(all_results), f, indent=2)

    print("\n" + "=" * 80)
    print("BASELINES + ABLATION TABLES (post-hoc, Community-SBM)")
    print("=" * 80)
    try:
        run_posthoc_baselines_and_ablation()
    except Exception as e:
        print(f"Post-hoc baselines/ablation failed: {e}")

    print(f"\nDone. Results: {results_file}")


if __name__ == "__main__":
    main()
