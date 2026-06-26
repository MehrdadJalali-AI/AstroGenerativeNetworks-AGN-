#!/usr/bin/env python3
"""
Plot generation for upgraded AGN experiments
Creates publication-ready diagnostic plots
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from agn_general.config import RESULTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_results(results_dir):
    """Load all results from JSON file"""
    results_file = Path(results_dir) / 'all_results.json'
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return []
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_edge_composition(results, output_dir):
    """Plot edge composition stacked bar charts"""
    output_dir = Path(output_dir) / 'figures' / 'global'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset and variant
    data = defaultdict(lambda: defaultdict(dict))
    
    for r in results:
        if not r.get('success'):
            continue
        dataset = r['dataset']
        variant = r['variant']
        if 'diagnostics' in r and 'edge_composition' in r['diagnostics']:
            ec = r['diagnostics']['edge_composition']
            data[dataset][variant] = {
                'orig_orig': ec.get('original_original', 0),
                'gen_orig': ec.get('generated_original', 0),
                'gen_gen': ec.get('generated_generated', 0)
            }
    
    # Create plot for each dataset
    for dataset, variants in data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variant_names = list(variants.keys())
        orig_orig = [variants[v].get('orig_orig', 0) for v in variant_names]
        gen_orig = [variants[v].get('gen_orig', 0) for v in variant_names]
        gen_gen = [variants[v].get('gen_gen', 0) for v in variant_names]
        
        x = np.arange(len(variant_names))
        width = 0.6
        
        ax.bar(x, orig_orig, width, label='Original-Original', color='#2E86AB')
        ax.bar(x, gen_orig, width, bottom=orig_orig, label='Generated-Original', color='#A23B72')
        ax.bar(x, gen_gen, width, bottom=np.array(orig_orig) + np.array(gen_orig),
               label='Generated-Generated', color='#F18F01')
        
        ax.set_xlabel('Variant', fontsize=12)
        ax.set_ylabel('Number of Edges', fontsize=12)
        ax.set_title(f'Edge Composition: {dataset}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'edge_composition_{dataset}.png')
        plt.savefig(output_dir / f'edge_composition_{dataset}.pdf')
        plt.close()


def plot_diagnostics_summary(results, output_dir):
    """Plot key diagnostic metrics"""
    output_dir = Path(output_dir) / 'figures' / 'global'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract key metrics
    metrics_data = []
    
    for r in results:
        if not r.get('success') or 'diagnostics' not in r:
            continue
        
        # Skip ablation results (only use main experiments)
        if r.get('ablation_type') is not None:
            continue
        
        d = r['diagnostics']
        metrics_data.append({
            'dataset': r['dataset'],
            'variant': r['variant'],
            'gg_edge_ratio': d.get('edge_composition', {}).get('gg_edge_ratio', 0),
            'isolated_gen_ratio': d.get('connectivity_analysis', {}).get('isolated_gen_ratio', 0),
            'dense_cluster_ratio': d.get('dense_cluster_warning', {}).get('max_gen_component_ratio', 0),
            'gen_avg_degree': d.get('degree_analysis', {}).get('generated_avg_degree', 0),
            'orig_avg_degree': d.get('degree_analysis', {}).get('original_avg_degree', 0)
        })
    
    if not metrics_data:
        print("No diagnostic data found")
        return
    
    df = pd.DataFrame(metrics_data)
    
    # Handle duplicates by taking mean (in case of multiple seeds)
    df_agg = df.groupby(['dataset', 'variant']).agg({
        'gg_edge_ratio': 'mean',
        'isolated_gen_ratio': 'mean',
        'dense_cluster_ratio': 'mean',
        'gen_avg_degree': 'mean',
        'orig_avg_degree': 'mean'
    }).reset_index()
    
    # Create heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GG edge ratio
    pivot_gg = df_agg.pivot(index='dataset', columns='variant', values='gg_edge_ratio')
    sns.heatmap(pivot_gg, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': 'Ratio'})
    axes[0, 0].set_title('Generated-Generated Edge Ratio', fontweight='bold')
    
    # Dense cluster ratio
    pivot_cluster = df_agg.pivot(index='dataset', columns='variant', values='dense_cluster_ratio')
    sns.heatmap(pivot_cluster, annot=True, fmt='.3f', cmap='Reds', ax=axes[0, 1], cbar_kws={'label': 'Ratio'})
    axes[0, 1].set_title('Max Generated Component Ratio', fontweight='bold')
    
    # Isolated nodes ratio
    pivot_isolated = df_agg.pivot(index='dataset', columns='variant', values='isolated_gen_ratio')
    sns.heatmap(pivot_isolated, annot=True, fmt='.3f', cmap='Blues', ax=axes[1, 0], cbar_kws={'label': 'Ratio'})
    axes[1, 0].set_title('Isolated Generated Nodes Ratio', fontweight='bold')
    
    # Degree ratio
    df_agg['degree_ratio'] = df_agg['gen_avg_degree'] / (df_agg['orig_avg_degree'] + 1e-8)
    pivot_degree = df_agg.pivot(index='dataset', columns='variant', values='degree_ratio')
    sns.heatmap(pivot_degree, annot=True, fmt='.2f', cmap='Greens', ax=axes[1, 1], cbar_kws={'label': 'Ratio'})
    axes[1, 1].set_title('Generated/Original Degree Ratio', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostics_summary_heatmap.png')
    plt.savefig(output_dir / 'diagnostics_summary_heatmap.pdf')
    plt.close()


def plot_novelty_comparison(results, output_dir):
    """Plot novelty metrics comparison"""
    output_dir = Path(output_dir) / 'figures' / 'global'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract novelty metrics
    novelty_data = []
    
    for r in results:
        if not r.get('success') or 'novelty' not in r:
            continue
        
        # Skip ablation results (only use main experiments)
        if r.get('ablation_type') is not None:
            continue
        
        n = r['novelty']
        novelty_data.append({
            'dataset': r['dataset'],
            'variant': r['variant'],
            'nearest_neighbor_mean': n.get('nearest_neighbor', {}).get('mean', 0),
            'mean_distance_to_orig': n.get('mean_distance_to_original', {}).get('mean', 0),
            'generated_diversity': n.get('generated_diversity', {}).get('mean_pairwise_distance', 0),
            'wasserstein_mean': n.get('wasserstein_distances', {}).get('mean', 0)
        })
    
    if not novelty_data:
        print("No novelty data found")
        return
    
    df = pd.DataFrame(novelty_data)
    
    # Handle duplicates by taking mean (in case of multiple seeds)
    df_agg = df.groupby(['dataset', 'variant']).agg({
        'nearest_neighbor_mean': 'mean',
        'mean_distance_to_orig': 'mean',
        'generated_diversity': 'mean',
        'wasserstein_mean': 'mean'
    }).reset_index()
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('nearest_neighbor_mean', 'Mean Nearest Neighbor Distance', axes[0, 0]),
        ('mean_distance_to_orig', 'Mean Distance to Original', axes[0, 1]),
        ('generated_diversity', 'Generated Diversity', axes[1, 0]),
        ('wasserstein_mean', 'Mean Wasserstein Distance', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        pivot = df_agg.pivot(index='dataset', columns='variant', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'novelty_comparison.png')
    plt.savefig(output_dir / 'novelty_comparison.pdf')
    plt.close()


def main():
    """Generate all plots"""
    results_dir = Path(RESULTS_DIR) / 'upgraded'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run experiments first using run_experiments_upgraded.py")
        return
    
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found")
        return
    
    print(f"Loaded {len(results)} results")
    print("Generating plots...")
    
    plot_edge_composition(results, results_dir)
    print("✓ Edge composition plots")
    
    plot_diagnostics_summary(results, results_dir)
    print("✓ Diagnostics summary plots")
    
    plot_novelty_comparison(results, results_dir)
    print("✓ Novelty comparison plots")
    
    print(f"\nAll plots saved to: {results_dir / 'figures'}")

    # Root-level IEEE figure bundle (PDF/PNG)
    import subprocess
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "build_paper_figures.py")],
            check=False,
        )
    except Exception as e:
        print(f"Paper figure export skipped: {e}")


if __name__ == "__main__":
    main()
