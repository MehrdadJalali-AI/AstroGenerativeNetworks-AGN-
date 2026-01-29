"""
Evaluation module for generated networks
Enhanced with comprehensive metrics and CSV exports
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import os
from .config import PLOTS_DIR, RESULTS_DIR

def compute_topology_metrics(G):
    """
    Compute comprehensive network topology metrics
    
    Returns:
        metrics: Dictionary of topology metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0
    metrics['median_degree'] = np.median(degrees) if degrees else 0
    metrics['std_degree'] = np.std(degrees) if degrees else 0
    metrics['min_degree'] = np.min(degrees) if degrees else 0
    metrics['max_degree'] = np.max(degrees) if degrees else 0
    
    # Connected components
    components = list(nx.connected_components(G))
    metrics['num_components'] = len(components)
    if components:
        component_sizes = [len(c) for c in components]
        metrics['giant_component_size'] = max(component_sizes)
        metrics['avg_component_size'] = np.mean(component_sizes)
        metrics['largest_component_ratio'] = max(component_sizes) / metrics['num_nodes']
    else:
        metrics['giant_component_size'] = 0
        metrics['avg_component_size'] = 0
        metrics['largest_component_ratio'] = 0
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    clustering_values = list(clustering.values())
    metrics['avg_clustering'] = np.mean(clustering_values) if clustering_values else 0
    metrics['std_clustering'] = np.std(clustering_values) if clustering_values else 0
    
    # Shortest path length (for largest connected component)
    if metrics['giant_component_size'] > 1:
        largest_component = max(components, key=len)
        G_lcc = G.subgraph(largest_component).copy()
        if G_lcc.number_of_nodes() > 1:
            try:
                path_lengths = dict(nx.all_pairs_shortest_path_length(G_lcc))
                all_lengths = []
                for source in path_lengths:
                    for target in path_lengths[source]:
                        if source != target:
                            all_lengths.append(path_lengths[source][target])
                metrics['avg_shortest_path_length'] = np.mean(all_lengths) if all_lengths else np.nan
                metrics['diameter'] = max(all_lengths) if all_lengths else np.nan
            except:
                metrics['avg_shortest_path_length'] = np.nan
                metrics['diameter'] = np.nan
        else:
            metrics['avg_shortest_path_length'] = np.nan
            metrics['diameter'] = np.nan
    else:
        metrics['avg_shortest_path_length'] = np.nan
        metrics['diameter'] = np.nan
    
    # Assortativity
    if metrics['num_nodes'] > 1 and metrics['num_edges'] > 0:
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            metrics['assortativity'] = np.nan
    else:
        metrics['assortativity'] = np.nan
    
    # Centrality measures (sample-based for large networks)
    if metrics['num_nodes'] <= 1000:
        # Compute for all nodes if network is small enough
        try:
            betweenness = nx.betweenness_centrality(G)
            metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
            metrics['max_betweenness'] = np.max(list(betweenness.values()))
        except:
            metrics['avg_betweenness'] = np.nan
            metrics['max_betweenness'] = np.nan
        
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            metrics['avg_eigenvector'] = np.mean(list(eigenvector.values()))
            metrics['max_eigenvector'] = np.max(list(eigenvector.values()))
        except:
            metrics['avg_eigenvector'] = np.nan
            metrics['max_eigenvector'] = np.nan
        
        try:
            pagerank = nx.pagerank(G)
            metrics['avg_pagerank'] = np.mean(list(pagerank.values()))
            metrics['max_pagerank'] = np.max(list(pagerank.values()))
        except:
            metrics['avg_pagerank'] = np.nan
            metrics['max_pagerank'] = np.nan
    else:
        # Sample for large networks
        sample_size = min(500, metrics['num_nodes'])
        sample_nodes = np.random.choice(list(G.nodes()), size=sample_size, replace=False)
        G_sample = G.subgraph(sample_nodes).copy()
        
        try:
            betweenness = nx.betweenness_centrality(G_sample)
            metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
            metrics['max_betweenness'] = np.max(list(betweenness.values()))
        except:
            metrics['avg_betweenness'] = np.nan
            metrics['max_betweenness'] = np.nan
        
        try:
            eigenvector = nx.eigenvector_centrality(G_sample, max_iter=1000)
            metrics['avg_eigenvector'] = np.mean(list(eigenvector.values()))
            metrics['max_eigenvector'] = np.max(list(eigenvector.values()))
        except:
            metrics['avg_eigenvector'] = np.nan
            metrics['max_eigenvector'] = np.nan
        
        try:
            pagerank = nx.pagerank(G_sample)
            metrics['avg_pagerank'] = np.mean(list(pagerank.values()))
            metrics['max_pagerank'] = np.max(list(pagerank.values()))
        except:
            metrics['avg_pagerank'] = np.nan
            metrics['max_pagerank'] = np.nan
    
    # Modularity (community detection)
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
        metrics['modularity'] = modularity
        metrics['num_communities'] = len(communities)
    except:
        metrics['modularity'] = np.nan
        metrics['num_communities'] = np.nan
    
    # Transitivity (global clustering)
    try:
        metrics['transitivity'] = nx.transitivity(G)
    except:
        metrics['transitivity'] = np.nan
    
    return metrics

def novelty_analysis(original_features, generated_features, epsilon=1e-6):
    """
    Analyze novelty of generated nodes compared to original
    Includes threshold-based and distributional metrics
    
    Args:
        original_features: Original node features
        generated_features: Generated node features
        epsilon: Threshold for near-duplicate detection
    
    Returns:
        results: Dictionary with comprehensive novelty metrics
    """
    # Compute pairwise distances
    all_features = np.vstack([original_features, generated_features])
    similarities = cosine_similarity(all_features)
    distances = 1 - similarities  # Convert similarity to distance
    
    n_original = len(original_features)
    n_generated = len(generated_features)
    
    # Distances from generated to original
    gen_to_orig = distances[n_original:, :n_original]
    min_distances = np.min(gen_to_orig, axis=1)
    mean_distances = np.mean(gen_to_orig, axis=1)
    max_distances = np.max(gen_to_orig, axis=1)
    
    # Distances within generated
    gen_to_gen = distances[n_original:, n_original:]
    np.fill_diagonal(gen_to_gen, np.inf)
    min_gen_distances = np.min(gen_to_gen, axis=1)
    mean_gen_distances = np.mean(gen_to_gen, axis=1)
    
    # Distributional metrics (NOT threshold-based)
    percentiles = [5, 25, 50, 75, 95]
    min_dist_percentiles = {f'p{p}': np.percentile(min_distances, p) for p in percentiles}
    mean_dist_percentiles = {f'p{p}': np.percentile(mean_distances, p) for p in percentiles}
    
    # Duplication rate (near-duplicates)
    near_duplicates = np.sum(min_distances < epsilon)
    duplication_rate = near_duplicates / n_generated
    
    # Novelty score (normalized)
    novelty_scores = min_distances / (np.mean(min_distances) + 1e-10)
    
    # Threshold-based classification (for consistency with previous work)
    threshold_25 = np.percentile(min_distances, 25)
    is_novel_threshold = (min_distances > threshold_25)
    novelty_percentage_threshold = np.mean(is_novel_threshold) * 100
    
    results = {
        # Distributional metrics (NOT threshold-based)
        'min_distance_mean': np.mean(min_distances),
        'min_distance_std': np.std(min_distances),
        'min_distance_min': np.min(min_distances),
        'min_distance_max': np.max(min_distances),
        'mean_distance_mean': np.mean(mean_distances),
        'mean_distance_std': np.std(mean_distances),
        'max_distance_mean': np.mean(max_distances),
        
        # Percentiles
        **min_dist_percentiles,
        **{f'mean_dist_{k}': v for k, v in mean_dist_percentiles.items()},
        
        # Duplication metrics
        'duplication_rate': duplication_rate,
        'near_duplicates': int(near_duplicates),
        
        # Generated-to-generated distances
        'min_gen_distance_mean': np.mean(min_gen_distances),
        'mean_gen_distance_mean': np.mean(mean_gen_distances),
        
        # Per-node metrics (for CSV export)
        'min_distance_to_original': min_distances.tolist(),
        'mean_distance_to_original': mean_distances.tolist(),
        'max_distance_to_original': max_distances.tolist(),
        'min_distance_to_generated': min_gen_distances.tolist(),
        'mean_distance_to_generated': mean_gen_distances.tolist(),
        'novelty_scores': novelty_scores.tolist(),
        
        # Threshold-based (explicitly labeled)
        'is_novel_threshold': is_novel_threshold.tolist(),
        'novelty_percentage_threshold': novelty_percentage_threshold,
        'threshold_value': threshold_25,
        
        # Summary statistics
        'avg_novelty_score': np.mean(novelty_scores),
        'std_novelty_score': np.std(novelty_scores),
    }
    
    return results

def save_metrics_to_csv(metrics_before, metrics_after, dataset_name, output_dir=RESULTS_DIR):
    """
    Save before/after metrics to CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Metric': list(metrics_before.keys()),
        'Before': [metrics_before[k] for k in metrics_before.keys()],
        'After': [metrics_after[k] for k in metrics_after.keys()]
    })
    
    # Calculate change
    df['Change'] = df['After'] - df['Before']
    df['Change_Percent'] = ((df['After'] - df['Before']) / (df['Before'].abs() + 1e-10)) * 100
    
    # Round numeric values
    numeric_cols = ['Before', 'After', 'Change', 'Change_Percent']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: round(x, 6) if pd.notna(x) and np.isfinite(x) else x)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'{dataset_name}_network_metrics_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics comparison saved to {csv_path}")
    
    return df

def plot_comparison(G_before, G_after, original_features, generated_features, 
                    dataset_name="network", output_dir=PLOTS_DIR):
    """
    Create comprehensive comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics for plotting
    metrics_before = compute_topology_metrics(G_before)
    metrics_after = compute_topology_metrics(G_after)
    
    # 1. Network visualization (sample if too large)
    max_viz_nodes = 500
    if G_before.number_of_nodes() > max_viz_nodes:
        sample_nodes_before = np.random.choice(list(G_before.nodes()), size=max_viz_nodes, replace=False)
        G_before_viz = G_before.subgraph(sample_nodes_before).copy()
    else:
        G_before_viz = G_before
    
    if G_after.number_of_nodes() > max_viz_nodes:
        original_nodes = [n for n in G_after.nodes() if n < G_before.number_of_nodes()]
        generated_nodes = [n for n in G_after.nodes() if n >= G_before.number_of_nodes()]
        # Keep all generated nodes and sample original
        n_orig_sample = max_viz_nodes - len(generated_nodes)
        if n_orig_sample > 0:
            sampled_orig = np.random.choice(original_nodes, size=min(n_orig_sample, len(original_nodes)), replace=False)
            sample_nodes_after = list(sampled_orig) + generated_nodes
        else:
            sample_nodes_after = generated_nodes[:max_viz_nodes]
        G_after_viz = G_after.subgraph(sample_nodes_after).copy()
    else:
        G_after_viz = G_after
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before
    pos_before = nx.spring_layout(G_before_viz, seed=42, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G_before_viz, pos_before, ax=axes[0], node_size=20, 
                           node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G_before_viz, pos_before, ax=axes[0], alpha=0.2, width=0.3)
    axes[0].set_title(f'Before AGN\n({G_before.number_of_nodes()} nodes, {G_before.number_of_edges()} edges)', fontsize=12)
    axes[0].axis('off')
    
    # After
    pos_after = nx.spring_layout(G_after_viz, seed=42, k=0.5, iterations=50)
    original_nodes_viz = [n for n in G_after_viz.nodes() if n < G_before.number_of_nodes()]
    generated_nodes_viz = [n for n in G_after_viz.nodes() if n >= G_before.number_of_nodes()]
    
    nx.draw_networkx_nodes(G_after_viz, pos_after, nodelist=original_nodes_viz, ax=axes[1],
                          node_size=20, node_color='blue', alpha=0.6)
    nx.draw_networkx_nodes(G_after_viz, pos_after, nodelist=generated_nodes_viz, ax=axes[1],
                          node_size=40, node_color='red', alpha=0.8)
    nx.draw_networkx_edges(G_after_viz, pos_after, ax=axes[1], alpha=0.2, width=0.3)
    axes[1].set_title(f'After AGN\n({G_after.number_of_nodes()} nodes, {G_after.number_of_edges()} edges)', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_network_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Degree distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    degrees_before = [d for n, d in G_before.degree()]
    degrees_after = [d for n, d in G_after.degree()]
    
    ax.hist(degrees_before, bins=50, alpha=0.6, label='Before', density=True, color='blue', edgecolor='black')
    ax.hist(degrees_after, bins=50, alpha=0.6, label='After', density=True, color='red', edgecolor='black')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Degree Distribution Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_degree_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics comparison bar chart
    key_metrics = ['num_nodes', 'num_edges', 'density', 'avg_degree', 'avg_clustering', 
                   'avg_shortest_path_length', 'assortativity', 'modularity']
    available_metrics = [m for m in key_metrics if m in metrics_before and pd.notna(metrics_before[m])]
    
    if available_metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(available_metrics))
        width = 0.35
        
        before_vals = [metrics_before[m] for m in available_metrics]
        after_vals = [metrics_after[m] for m in available_metrics]
        
        # Normalize for visualization
        max_vals = [max(abs(b), abs(a)) for b, a in zip(before_vals, after_vals)]
        before_norm = [b / (m + 1e-10) for b, m in zip(before_vals, max_vals)]
        after_norm = [a / (m + 1e-10) for a, m in zip(after_vals, max_vals)]
        
        ax.bar(x - width/2, before_norm, width, label='Before', color='blue', alpha=0.7)
        ax.bar(x + width/2, after_norm, width, label='After', color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title('Network Metrics Comparison (Normalized)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_metrics_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. PCA visualization
    pca = PCA(n_components=2)
    all_features_combined = np.vstack([original_features, generated_features])
    pca_result = pca.fit_transform(all_features_combined)
    
    n_orig = len(original_features)
    n_gen = len(generated_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_result[:n_orig, 0], pca_result[:n_orig, 1],
              alpha=0.5, label='Original', s=20, color='blue')
    ax.scatter(pca_result[n_orig:, 0], pca_result[n_orig:, 1],
              alpha=0.8, label='Generated', s=40, color='red', marker='^')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA: Original vs Generated Nodes', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_pca.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Novelty analysis
    novelty_results = novelty_analysis(original_features, generated_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Min distance histogram
    min_distances = novelty_results.get('min_distance_to_original', [])
    if len(min_distances) > 0:
        axes[0].hist(min_distances, bins=30, 
                    alpha=0.7, color='green', edgecolor='black')
        mean_min_dist = np.mean(min_distances)
        axes[0].axvline(mean_min_dist, 
                       color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_min_dist:.3f}')
    axes[0].set_xlabel('Minimum Distance to Original Nodes', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Novelty: Distance to Nearest Original Node', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Novelty scores
    is_novel = novelty_results.get('is_novel_threshold', novelty_results.get('is_novel', []))
    novelty_pct = novelty_results.get('novelty_percentage_threshold', novelty_results.get('novelty_percentage', 0.0))
    novelty_scores = novelty_results.get('novelty_scores', [])
    if len(novelty_scores) > 0:
        axes[1].bar(range(len(novelty_scores)), 
                   novelty_scores,
                   color=['green' if novel else 'orange' 
                         for novel in is_novel] if len(is_novel) == len(novelty_scores) else 'blue',
                   edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Generated Node Index', fontsize=11)
    axes[1].set_ylabel('Novelty Score', fontsize=11)
    axes[1].set_title(f'Novelty Scores (Novel: {novelty_pct:.1f}%)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_novelty.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}")
    return novelty_results

def generate_evaluation_report(G_before, G_after, original_features, generated_features,
                               dataset_name="network", output_dir=PLOTS_DIR):
    """
    Generate comprehensive evaluation report with CSV export
    """
    # Compute metrics
    metrics_before = compute_topology_metrics(G_before)
    metrics_after = compute_topology_metrics(G_after)
    
    # Save metrics to CSV
    metrics_df = save_metrics_to_csv(metrics_before, metrics_after, dataset_name)
    
    # Novelty analysis
    novelty_results = novelty_analysis(original_features, generated_features)
    
    # Save novelty results to CSV
    is_novel = novelty_results.get('is_novel_threshold', novelty_results.get('is_novel', []))
    # Use .get() with defaults for all keys to handle missing values gracefully
    novelty_df = pd.DataFrame({
        'Node_Index': range(len(generated_features)),
        'Min_Distance_to_Original': novelty_results.get('min_distance_to_original', []),
        'Mean_Distance_to_Original': novelty_results.get('mean_distance_to_original', []),
        'Max_Distance_to_Original': novelty_results.get('max_distance_to_original', []),
        'Min_Distance_to_Generated': novelty_results.get('min_distance_to_generated', []),
        'Mean_Distance_to_Generated': novelty_results.get('mean_distance_to_generated', []),
        'Novelty_Score': novelty_results.get('novelty_scores', []),
        'Is_Novel': is_novel
    })
    novelty_csv_path = os.path.join(RESULTS_DIR, f'{dataset_name}_novelty_analysis.csv')
    novelty_df.to_csv(novelty_csv_path, index=False)
    print(f"Novelty analysis saved to {novelty_csv_path}")
    
    # Create plots
    plot_comparison(G_before, G_after, original_features, generated_features, 
                   dataset_name, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print(f"EVALUATION REPORT: {dataset_name.upper()}")
    print("="*80)
    print("\nTopology Metrics:")
    print(f"  Nodes: {metrics_before['num_nodes']} -> {metrics_after['num_nodes']} "
          f"(+{metrics_after['num_nodes'] - metrics_before['num_nodes']})")
    print(f"  Edges: {metrics_before['num_edges']} -> {metrics_after['num_edges']} "
          f"(+{metrics_after['num_edges'] - metrics_before['num_edges']})")
    print(f"  Density: {metrics_before['density']:.6f} -> {metrics_after['density']:.6f}")
    print(f"  Avg Degree: {metrics_before['avg_degree']:.2f} -> {metrics_after['avg_degree']:.2f}")
    print(f"  Clustering: {metrics_before['avg_clustering']:.4f} -> {metrics_after['avg_clustering']:.4f}")
    if pd.notna(metrics_before['avg_shortest_path_length']):
        print(f"  Avg Path Length: {metrics_before['avg_shortest_path_length']:.4f} -> "
              f"{metrics_after['avg_shortest_path_length']:.4f}")
    if pd.notna(metrics_before['modularity']):
        print(f"  Modularity: {metrics_before['modularity']:.4f} -> {metrics_after['modularity']:.4f}")
    if pd.notna(metrics_before['assortativity']):
        print(f"  Assortativity: {metrics_before['assortativity']:.4f} -> {metrics_after['assortativity']:.4f}")
    
    print("\nNovelty Analysis:")
    avg_novelty = novelty_results.get('avg_novelty_score', novelty_results.get('avg_novelty', 0.0))
    std_novelty = novelty_results.get('std_novelty_score', novelty_results.get('std_novelty', 0.0))
    novelty_pct = novelty_results.get('novelty_percentage_threshold', novelty_results.get('novelty_percentage', 0.0))
    min_dist_list = novelty_results.get('min_distance_to_original', [])
    min_dist_mean = novelty_results.get('min_distance_mean', np.mean(min_dist_list) if len(min_dist_list) > 0 else 0.0)
    print(f"  Average Novelty Score: {avg_novelty:.4f} ± {std_novelty:.4f}")
    print(f"  Novelty Percentage (threshold-based): {novelty_pct:.1f}%")
    print(f"  Avg Min Distance to Original: {min_dist_mean:.4f}")
    if 'mean_distance_mean' in novelty_results:
        print(f"  Avg Mean Distance to Original: {novelty_results.get('mean_distance_mean', 0.0):.4f}")
    if 'duplication_rate' in novelty_results:
        print(f"  Duplication Rate: {novelty_results.get('duplication_rate', 0.0):.4f}")
    
    print("="*80 + "\n")
    
    # Ensure backward compatibility
    novelty_results_compat = novelty_results.copy()
    if 'is_novel_threshold' in novelty_results_compat and 'is_novel' not in novelty_results_compat:
        novelty_results_compat['is_novel'] = novelty_results_compat['is_novel_threshold']
    if 'novelty_percentage_threshold' in novelty_results_compat and 'novelty_percentage' not in novelty_results_compat:
        novelty_results_compat['novelty_percentage'] = novelty_results_compat['novelty_percentage_threshold']
    if 'avg_novelty_score' in novelty_results_compat and 'avg_novelty' not in novelty_results_compat:
        novelty_results_compat['avg_novelty'] = novelty_results_compat['avg_novelty_score']
    if 'std_novelty_score' in novelty_results_compat and 'std_novelty' not in novelty_results_compat:
        novelty_results_compat['std_novelty'] = novelty_results_compat['std_novelty_score']
    
    return {
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'novelty': novelty_results_compat,
        'metrics_df': metrics_df,
        'novelty_df': novelty_df
    }
