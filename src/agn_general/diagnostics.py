"""
Comprehensive diagnostics for node insertion evaluation
Addresses reviewer concerns about sparse-graph failure modes
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import json


def compute_insertion_diagnostics(G_original, G_augmented, original_features, generated_features,
                                  generated_node_ids):
    """
    Compute comprehensive insertion diagnostics
    
    Returns:
        diagnostics: Dictionary with all diagnostic metrics
    """
    diagnostics = {}
    
    n_original = G_original.number_of_nodes()
    n_generated = len(generated_node_ids)
    
    # Edge composition analysis
    original_edges = set(G_original.edges())
    all_edges = set(G_augmented.edges())
    new_edges = all_edges - original_edges
    
    # Count edge types
    orig_orig_edges = 0
    gen_orig_edges = 0
    gen_gen_edges = 0
    
    for u, v in new_edges:
        u_is_gen = u in generated_node_ids
        v_is_gen = v in generated_node_ids
        
        if u_is_gen and v_is_gen:
            gen_gen_edges += 1
        elif u_is_gen or v_is_gen:
            gen_orig_edges += 1
        else:
            orig_orig_edges += 1
    
    diagnostics['edge_composition'] = {
        'original_original': orig_orig_edges,
        'generated_original': gen_orig_edges,
        'generated_generated': gen_gen_edges,
        'total_new_edges': len(new_edges),
        'gg_edge_ratio': gen_gen_edges / len(new_edges) if len(new_edges) > 0 else 0.0
    }
    
    # Generated node degree analysis
    gen_degrees = [G_augmented.degree(n) for n in generated_node_ids]
    orig_degrees = [G_augmented.degree(n) for n in range(n_original)]
    
    diagnostics['degree_analysis'] = {
        'generated_avg_degree': np.mean(gen_degrees) if gen_degrees else 0.0,
        'generated_median_degree': np.median(gen_degrees) if gen_degrees else 0.0,
        'generated_std_degree': np.std(gen_degrees) if gen_degrees else 0.0,
        'original_avg_degree': np.mean(orig_degrees) if orig_degrees else 0.0,
        'original_median_degree': np.median(orig_degrees) if orig_degrees else 0.0,
        'original_std_degree': np.std(orig_degrees) if orig_degrees else 0.0,
        'degree_ratio': np.mean(gen_degrees) / np.mean(orig_degrees) if orig_degrees and np.mean(orig_degrees) > 0 else 0.0
    }
    
    # Generated node connectivity to original nodes
    gen_to_orig_counts = []
    gen_to_gen_counts = []
    isolated_gen_nodes = 0
    
    for gen_node in generated_node_ids:
        neighbors = list(G_augmented.neighbors(gen_node))
        orig_neighbors = [n for n in neighbors if n < n_original]
        gen_neighbors = [n for n in neighbors if n in generated_node_ids]
        
        gen_to_orig_counts.append(len(orig_neighbors))
        gen_to_gen_counts.append(len(gen_neighbors))
        
        if len(orig_neighbors) == 0:
            isolated_gen_nodes += 1
    
    diagnostics['connectivity_analysis'] = {
        'avg_gen_to_orig_edges': np.mean(gen_to_orig_counts) if gen_to_orig_counts else 0.0,
        'avg_gen_to_gen_edges': np.mean(gen_to_gen_counts) if gen_to_gen_counts else 0.0,
        'isolated_gen_nodes': isolated_gen_nodes,
        'isolated_gen_ratio': isolated_gen_nodes / n_generated if n_generated > 0 else 0.0,
        'gen_nodes_mostly_to_gen': sum(1 for i, (orig_c, gen_c) in enumerate(zip(gen_to_orig_counts, gen_to_gen_counts))
                                       if gen_c > orig_c) if gen_to_orig_counts else 0
    }
    
    # Component analysis
    components = list(nx.connected_components(G_augmented))
    gen_component_sizes = []
    gen_in_largest_component = 0
    
    if components:
        largest_component = max(components, key=len)
        for comp in components:
            gen_nodes_in_comp = [n for n in comp if n in generated_node_ids]
            gen_component_sizes.append(len(gen_nodes_in_comp))
            if comp == largest_component:
                gen_in_largest_component = len(gen_nodes_in_comp)
    
    diagnostics['component_analysis'] = {
        'num_components': len(components),
        'largest_component_size': len(max(components, key=len)) if components else 0,
        'gen_nodes_in_largest_component': gen_in_largest_component,
        'gen_in_largest_component_ratio': gen_in_largest_component / n_generated if n_generated > 0 else 0.0,
        'max_gen_component_size': max(gen_component_sizes) if gen_component_sizes else 0,
        'gen_component_sizes': gen_component_sizes[:10]  # Store first 10 for inspection
    }
    
    # Check for dense generated cluster
    if gen_component_sizes:
        max_gen_comp = max(gen_component_sizes)
        diagnostics['dense_cluster_warning'] = {
            'has_large_gen_component': max_gen_comp > n_generated * 0.5,
            'max_gen_component_ratio': max_gen_comp / n_generated if n_generated > 0 else 0.0
        }
    else:
        diagnostics['dense_cluster_warning'] = {
            'has_large_gen_component': False,
            'max_gen_component_ratio': 0.0
        }
    
    # Clustering coefficient for generated nodes
    gen_clustering = [nx.clustering(G_augmented, n) for n in generated_node_ids]
    orig_clustering = [nx.clustering(G_augmented, n) for n in range(n_original)]
    
    diagnostics['clustering_analysis'] = {
        'generated_avg_clustering': np.mean(gen_clustering) if gen_clustering else 0.0,
        'original_avg_clustering': np.mean(orig_clustering) if orig_clustering else 0.0,
        'clustering_ratio': np.mean(gen_clustering) / np.mean(orig_clustering) if orig_clustering and np.mean(orig_clustering) > 0 else 0.0
    }
    
    # Similarity distributions
    orig_norm = original_features / (np.linalg.norm(original_features, axis=1, keepdims=True) + 1e-8)
    gen_norm = generated_features / (np.linalg.norm(generated_features, axis=1, keepdims=True) + 1e-8)
    
    gen_to_orig_sim = cosine_similarity(gen_norm, orig_norm)
    gen_to_gen_sim = cosine_similarity(gen_norm, gen_norm)
    orig_to_orig_sim = cosine_similarity(orig_norm, orig_norm)
    
    # Remove diagonal from orig_to_orig
    orig_to_orig_sim_flat = orig_to_orig_sim[np.triu_indices_from(orig_to_orig_sim, k=1)]
    
    diagnostics['similarity_distributions'] = {
        'gen_to_orig_mean': np.mean(gen_to_orig_sim),
        'gen_to_orig_std': np.std(gen_to_orig_sim),
        'gen_to_orig_min': np.min(gen_to_orig_sim),
        'gen_to_orig_max': np.max(gen_to_orig_sim),
        'gen_to_gen_mean': np.mean(gen_to_gen_sim[np.triu_indices_from(gen_to_gen_sim, k=1)]),
        'gen_to_gen_std': np.std(gen_to_gen_sim[np.triu_indices_from(gen_to_gen_sim, k=1)]),
        'orig_to_orig_mean': np.mean(orig_to_orig_sim_flat),
        'orig_to_orig_std': np.std(orig_to_orig_sim_flat)
    }
    
    # Nearest neighbor analysis
    nearest_orig_distances = []
    nearest_gen_distances = []
    
    for i, gen_feat in enumerate(gen_norm):
        # Distance to nearest original node
        dists_to_orig = 1 - gen_to_orig_sim[i]  # Convert similarity to distance
        nearest_orig_distances.append(np.min(dists_to_orig))
        
        # Distance to nearest generated node (excluding self)
        dists_to_gen = 1 - gen_to_gen_sim[i]
        dists_to_gen[i] = np.inf  # Exclude self
        nearest_gen_distances.append(np.min(dists_to_gen))
    
    diagnostics['nearest_neighbor_analysis'] = {
        'mean_nearest_orig_distance': np.mean(nearest_orig_distances) if nearest_orig_distances else 0.0,
        'std_nearest_orig_distance': np.std(nearest_orig_distances) if nearest_orig_distances else 0.0,
        'mean_nearest_gen_distance': np.mean(nearest_gen_distances) if nearest_gen_distances else 0.0,
        'std_nearest_gen_distance': np.std(nearest_gen_distances) if nearest_gen_distances else 0.0,
        'nearest_orig_vs_gen_ratio': np.mean(nearest_orig_distances) / np.mean(nearest_gen_distances) if nearest_gen_distances and np.mean(nearest_gen_distances) > 0 else 0.0
    }
    
    # Topology metrics before/after
    orig_density = nx.density(G_original)
    aug_density = nx.density(G_augmented)
    
    orig_assort = nx.degree_assortativity_coefficient(G_original) if G_original.number_of_edges() > 0 else 0.0
    aug_assort = nx.degree_assortativity_coefficient(G_augmented) if G_augmented.number_of_edges() > 0 else 0.0
    
    try:
        orig_communities = nx.community.greedy_modularity_communities(G_original)
        orig_modularity = nx.community.modularity(G_original, orig_communities)
    except:
        orig_modularity = 0.0
    
    try:
        aug_communities = nx.community.greedy_modularity_communities(G_augmented)
        aug_modularity = nx.community.modularity(G_augmented, aug_communities)
    except:
        aug_modularity = 0.0
    
    diagnostics['topology_changes'] = {
        'density_before': orig_density,
        'density_after': aug_density,
        'density_change': aug_density - orig_density,
        'density_change_pct': ((aug_density - orig_density) / orig_density * 100) if orig_density > 0 else 0.0,
        'assortativity_before': orig_assort,
        'assortativity_after': aug_assort,
        'assortativity_change': aug_assort - orig_assort,
        'modularity_before': orig_modularity,
        'modularity_after': aug_modularity,
        'modularity_change': aug_modularity - orig_modularity
    }
    
    return diagnostics


def save_diagnostics(diagnostics, dataset_name, method_name, output_dir):
    """Save diagnostics to JSON file"""
    import os
    os.makedirs(os.path.join(output_dir, dataset_name, method_name), exist_ok=True)
    filepath = os.path.join(output_dir, dataset_name, method_name, 'diagnostics.json')
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    diagnostics_serializable = convert_to_native(diagnostics)
    
    with open(filepath, 'w') as f:
        json.dump(diagnostics_serializable, f, indent=2)
    
    return filepath
