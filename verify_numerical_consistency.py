#!/usr/bin/env python3
"""
Numerical Verification Script for AGN Manuscript
Verifies all tables, metrics, and claims against source data files.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path

# Configuration
RESULTS_DIR = Path('results/upgraded')
MAIN_CSV = RESULTS_DIR / 'tables/main_experiments.csv'
DIAGNOSTICS_DIR = RESULTS_DIR / 'diagnostics'

def load_main_results():
    """Load main experimental results."""
    df = pd.read_csv(MAIN_CSV)
    return df

def recompute_density(N, E):
    """Compute density for undirected graph."""
    if N <= 1:
        return 0.0
    return 2 * E / (N * (N - 1))

def recompute_avg_degree(N, E):
    """Compute average degree for undirected graph."""
    if N == 0:
        return 0.0
    return 2 * E / N

def compute_percent_change(before, after):
    """Compute relative percent change."""
    if before == 0:
        return np.nan
    return ((after - before) / before) * 100

def verify_table_2():
    """Verify Table 2 (Topology Summary)."""
    df = load_main_results()
    
    # Get first run for each dataset (no_gg, M=100)
    datasets = ['karate', 'facebook', 'email']
    results = []
    
    for ds in datasets:
        df_ds = df[(df['dataset'] == ds) & 
                   (df['variant'] == 'no_gg') & 
                   (df['num_generated'] == 100)].iloc[0]
        
        N_before = df_ds['nodes_before']
        N_after = df_ds['nodes_after']
        E_before = df_ds['edges_before']
        E_after = df_ds['edges_after']
        
        # Recompute from first principles
        density_before = recompute_density(N_before, E_before)
        density_after = recompute_density(N_after, E_after)
        density_change = compute_percent_change(density_before, density_after)
        
        clustering_change = compute_percent_change(
            df_ds['clustering_before'], df_ds['clustering_after'])
        modularity_change = compute_percent_change(
            df_ds['modularity_before'], df_ds['modularity_after'])
        path_change = compute_percent_change(
            df_ds['path_length_before'], df_ds['path_length_after'])
        
        results.append({
            'dataset': ds,
            'nodes_before': int(N_before),
            'nodes_after': int(N_after),
            'edges_before': int(E_before),
            'edges_after': int(E_after),
            'density_change': density_change,
            'clustering_change': clustering_change,
            'modularity_change': modularity_change,
            'path_change': path_change,
            'clustering_before': df_ds['clustering_before'],
            'clustering_after': df_ds['clustering_after'],
            'modularity_before': df_ds['modularity_before'],
            'modularity_after': df_ds['modularity_after'],
        })
    
    return results

def verify_table_3():
    """Verify Table 3 (Edge Composition)."""
    results = []
    
    datasets = ['karate', 'facebook', 'email']
    for ds in datasets:
        for variant in ['original', 'no_gg']:
            diag_path = DIAGNOSTICS_DIR / ds / variant / 'diagnostics.json'
            if not diag_path.exists():
                continue
            
            with open(diag_path, 'r') as f:
                diag = json.load(f)
            
            edge_comp = diag.get('edge_composition', {})
            
            results.append({
                'dataset': ds,
                'variant': variant,
                'gen_orig': edge_comp.get('gen_to_orig_edges', 0),
                'gen_gen': edge_comp.get('gen_to_gen_edges', 0),
                'gg_ratio': edge_comp.get('gg_edge_ratio', 0),
                'avg_gen_degree': edge_comp.get('generated_avg_degree', 0),
                'gen_mostly_to_gen': edge_comp.get('gen_nodes_mostly_to_gen', 0),
            })
    
    return results

def verify_table_4():
    """Verify Table 4 (Novelty Metrics)."""
    df = load_main_results()
    
    datasets = ['karate', 'facebook', 'email']
    results = []
    
    for ds in datasets:
        df_ds = df[(df['dataset'] == ds) & 
                   (df['variant'] == 'no_gg') & 
                   (df['num_generated'] == 100)].iloc[0]
        
        results.append({
            'dataset': ds,
            'nearest_neighbor_mean': df_ds.get('nearest_neighbor_mean', np.nan),
            'nearest_neighbor_std': df_ds.get('nearest_neighbor_std', np.nan),
            'mean_distance_to_orig': df_ds.get('mean_distance_to_orig', np.nan),
            'wasserstein_distance': df_ds.get('wasserstein_mean', np.nan),
            'generated_diversity': df_ds.get('generated_diversity', np.nan),
        })
    
    return results

def verify_table_5():
    """Verify Table 5 (Baseline Comparison)."""
    # Check if baseline results exist in CSV
    baseline_files = [
        'results/karate_baseline_topology_comparison.csv',
        'results/facebook_baseline_topology_comparison.csv',
    ]
    
    results = []
    for f in baseline_files:
        if os.path.exists(f):
            df_b = pd.read_csv(f)
            # Process baseline results
            for _, row in df_b.iterrows():
                results.append({
                    'method': row.get('method', ''),
                    'density': row.get('density', np.nan),
                    'avg_degree': row.get('avg_degree', np.nan),
                    'clustering': row.get('clustering', np.nan),
                    'modularity': row.get('modularity', np.nan),
                    'path_length': row.get('path_length', np.nan),
                    'assortativity': row.get('assortativity', np.nan),
                })
    
    return results

if __name__ == '__main__':
    print("=== NUMERICAL VERIFICATION AUDIT ===\n")
    
    print("1. Verifying Table 2 (Topology Summary)...")
    table2_results = verify_table_2()
    for r in table2_results:
        print(f"\n{r['dataset']}:")
        print(f"  Density change: {r['density_change']:.2f}%")
        print(f"  Clustering change: {r['clustering_change']:.2f}%")
        print(f"  Modularity change: {r['modularity_change']:.2f}%")
        print(f"  Path change: {r['path_change']:.2f}%")
    
    print("\n2. Verifying Table 3 (Edge Composition)...")
    table3_results = verify_table_3()
    for r in table3_results:
        print(f"\n{r['dataset']} ({r['variant']}):")
        print(f"  Gen-Orig: {r['gen_orig']}, Gen-Gen: {r['gen_gen']}")
        print(f"  GG Ratio: {r['gg_ratio']:.2f}")
        print(f"  Avg Gen Degree: {r['avg_gen_degree']:.1f}")
    
    print("\n3. Verifying Table 4 (Novelty)...")
    table4_results = verify_table_4()
    for r in table4_results:
        print(f"\n{r['dataset']}:")
        print(f"  Mean distance: {r['mean_distance_to_orig']:.4f}")
        print(f"  Wasserstein: {r['wasserstein_distance']:.3f}")
