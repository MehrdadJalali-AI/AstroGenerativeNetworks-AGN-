#!/usr/bin/env python3
"""
Aggregate experimental results into CSV tables for manuscript
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from agn_general.config import RESULTS_DIR


def load_results(results_dir):
    """Load all results"""
    results_file = Path(results_dir) / 'all_results.json'
    if not results_file.exists():
        return []
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_main_results_table(results):
    """Create main results CSV table"""
    rows = []
    
    for r in results:
        if not r.get('success'):
            continue
        
        row = {
            'dataset': r['dataset'],
            'variant': r['variant'],
            'num_generated': r.get('num_generated', 0),
            'k': r.get('k', 0),
            'tau': r.get('tau', 0),
            'ablation_type': r.get('ablation_type'),
            'ablation_value': r.get('ablation_value'),
        }
        
        # Topology metrics
        if 'topology' in r:
            orig = r['topology']['original']
            aug = r['topology']['augmented']
            changes = r['topology']['changes']
            
            row.update({
                'nodes_before': orig.get('num_nodes', 0),
                'nodes_after': aug.get('num_nodes', 0),
                'edges_before': orig.get('num_edges', 0),
                'edges_after': aug.get('num_edges', 0),
                'density_before': orig.get('density', 0),
                'density_after': aug.get('density', 0),
                'density_change': changes.get('density', 0),
                'clustering_before': orig.get('avg_clustering', 0),
                'clustering_after': aug.get('avg_clustering', 0),
                'clustering_change': changes.get('clustering', 0),
                'modularity_before': orig.get('modularity', 0),
                'modularity_after': aug.get('modularity', 0),
                'modularity_change': changes.get('modularity', 0),
                'path_length_before': orig.get('avg_shortest_path_length', 0),
                'path_length_after': aug.get('avg_shortest_path_length', 0),
                'path_length_change': changes.get('path_length', 0),
                'assortativity_before': orig.get('assortativity', 0),
                'assortativity_after': aug.get('assortativity', 0),
                'assortativity_change': changes.get('assortativity', 0),
            })
        
        # Diagnostics
        if 'diagnostics' in r:
            d = r['diagnostics']
            ec = d.get('edge_composition', {})
            ca = d.get('connectivity_analysis', {})
            dcw = d.get('dense_cluster_warning', {})
            
            row.update({
                'gen_orig_edges': ec.get('generated_original', 0),
                'gen_gen_edges': ec.get('generated_generated', 0),
                'gg_edge_ratio': ec.get('gg_edge_ratio', 0),
                'isolated_gen_nodes': ca.get('isolated_gen_nodes', 0),
                'isolated_gen_ratio': ca.get('isolated_gen_ratio', 0),
                'has_dense_cluster': dcw.get('has_large_gen_component', False),
                'max_gen_component_ratio': dcw.get('max_gen_component_ratio', 0),
            })
        
        # Novelty
        if 'novelty' in r:
            n = r['novelty']
            row.update({
                'nearest_neighbor_mean': n.get('nearest_neighbor', {}).get('mean', 0),
                'nearest_neighbor_std': n.get('nearest_neighbor', {}).get('std', 0),
                'mean_distance_to_orig': n.get('mean_distance_to_original', {}).get('mean', 0),
                'generated_diversity': n.get('generated_diversity', {}).get('mean_pairwise_distance', 0),
                'wasserstein_mean': n.get('wasserstein_distances', {}).get('mean', 0),
            })
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_summary_statistics(df):
    """Create summary statistics table"""
    if df.empty:
        return pd.DataFrame()
    
    # Group by dataset and variant
    summary_cols = [
        'density_change', 'clustering_change', 'modularity_change',
        'gg_edge_ratio', 'isolated_gen_ratio', 'nearest_neighbor_mean'
    ]
    
    summary = df.groupby(['dataset', 'variant'])[summary_cols].agg(['mean', 'std', 'count'])
    
    return summary


def main():
    """Aggregate results into tables"""
    results_dir = Path(RESULTS_DIR) / 'upgraded'
    tables_dir = results_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found")
        return
    
    print(f"Loaded {len(results)} results")
    print("Creating tables...")
    
    # Main results table
    df_main = create_main_results_table(results)
    df_main.to_csv(tables_dir / 'main_results.csv', index=False)
    print(f"✓ Main results table: {len(df_main)} rows")
    
    # Summary statistics
    df_summary = create_summary_statistics(df_main)
    df_summary.to_csv(tables_dir / 'summary_statistics.csv')
    print("✓ Summary statistics table")
    
    # Main grid: exclude hyperparameter ablation rows
    if 'ablation_type' in df_main.columns:
        df_main_experiments = df_main[df_main['ablation_type'].isna() | (df_main['ablation_type'] == '')]
    else:
        df_main_experiments = df_main
    df_main_experiments.to_csv(tables_dir / 'main_experiments.csv', index=False)
    print(f"✓ Main experiments table: {len(df_main_experiments)} rows")
    
    print(f"\nAll tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
