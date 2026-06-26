"""
Generation variants for AGN with different insertion strategies
Addresses sparse-graph failure modes and generated-generated edge issues
"""

import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from .config import DEVICE


def normalize_features(feat):
    """Normalize features for similarity computation"""
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feat / norms


def compute_density_adaptive_params(G, base_k=10, base_tau=0.5):
    """
    Adapt k and tau based on graph density
    
    Args:
        G: NetworkX graph
        base_k: Base k value
        base_tau: Base threshold value
    
    Returns:
        k_adapted: Adapted k value
        tau_adapted: Adapted threshold value
    """
    density = nx.density(G)
    avg_degree = np.mean([d for _, d in G.degree()])
    
    # Adapt k: sparse graphs need fewer connections
    if density < 0.01:  # Very sparse
        k_adapted = max(3, int(base_k * 0.5))
        tau_adapted = base_tau * 0.8  # Lower threshold for sparse graphs
    elif density < 0.05:  # Sparse
        k_adapted = max(5, int(base_k * 0.7))
        tau_adapted = base_tau * 0.9
    elif density > 0.3:  # Dense
        k_adapted = int(base_k * 1.2)
        tau_adapted = base_tau * 1.1
    else:  # Moderate
        k_adapted = base_k
        tau_adapted = base_tau
    
    return k_adapted, tau_adapted


def insert_nodes_original(G_original, original_features, generated_features,
                         k_neighbors=10, threshold=0.5):
    """
    Original AGN insertion logic (baseline)
    - Connects generated nodes to top-k original nodes
    - Connects generated nodes to each other if similarity >= threshold
    """
    G_augmented = G_original.copy()
    n_original = len(G_original.nodes())
    n_generated = len(generated_features)
    
    orig_norm = normalize_features(original_features)
    gen_norm = normalize_features(generated_features)
    
    similarities = cosine_similarity(gen_norm, orig_norm)
    
    generated_node_ids = []
    for i in range(n_generated):
        new_node_id = n_original + i
        generated_node_ids.append(new_node_id)
        G_augmented.add_node(new_node_id, is_generated=True)
        
        # Connect to top-k nearest original nodes
        top_k_indices = np.argsort(similarities[i])[-k_neighbors:]
        for orig_idx in top_k_indices:
            sim_value = similarities[i, orig_idx]
            if sim_value >= threshold:
                G_augmented.add_edge(new_node_id, orig_idx, weight=sim_value, 
                                   edge_type='gen_orig', is_generated=True)
    
    # Connect generated nodes to each other (unrestricted)
    gen_similarities = cosine_similarity(gen_norm, gen_norm)
    for i in range(n_generated):
        for j in range(i + 1, n_generated):
            sim_value = gen_similarities[i, j]
            if sim_value >= threshold:
                node_i = n_original + i
                node_j = n_original + j
                if not G_augmented.has_edge(node_i, node_j):
                    G_augmented.add_edge(node_i, node_j, weight=sim_value,
                                       edge_type='gen_gen', is_generated=True)
    
    return G_augmented, generated_node_ids


def insert_nodes_no_gg(G_original, original_features, generated_features,
                       k_neighbors=10, threshold=0.5):
    """
    AGN variant: NO generated-generated edges
    - Only connects generated nodes to original nodes
    """
    G_augmented = G_original.copy()
    n_original = len(G_original.nodes())
    n_generated = len(generated_features)
    
    orig_norm = normalize_features(original_features)
    gen_norm = normalize_features(generated_features)
    
    similarities = cosine_similarity(gen_norm, orig_norm)
    
    generated_node_ids = []
    for i in range(n_generated):
        new_node_id = n_original + i
        generated_node_ids.append(new_node_id)
        G_augmented.add_node(new_node_id, is_generated=True)
        
        # Connect to top-k nearest original nodes
        top_k_indices = np.argsort(similarities[i])[-k_neighbors:]
        for orig_idx in top_k_indices:
            sim_value = similarities[i, orig_idx]
            if sim_value >= threshold:
                G_augmented.add_edge(new_node_id, orig_idx, weight=sim_value,
                                   edge_type='gen_orig', is_generated=True)
    
    # NO generated-generated edges
    
    return G_augmented, generated_node_ids


def insert_nodes_strict_gg(G_original, original_features, generated_features,
                          k_neighbors=10, threshold=0.5, tau_gg_multiplier=1.2,
                          max_gg_edges_per_node=3):
    """
    AGN variant: STRICT generated-generated edges
    - Mutual top-k condition: both nodes must be in each other's top-k generated neighbors
    - Higher threshold: tau_gg = tau * tau_gg_multiplier
    - Hard cap: max_gg_edges_per_node generated-generated edges per node
    """
    G_augmented = G_original.copy()
    n_original = len(G_original.nodes())
    n_generated = len(generated_features)
    
    orig_norm = normalize_features(original_features)
    gen_norm = normalize_features(generated_features)
    
    similarities = cosine_similarity(gen_norm, orig_norm)
    gen_similarities = cosine_similarity(gen_norm, gen_norm)
    
    # Higher threshold for generated-generated edges
    threshold_gg = threshold * tau_gg_multiplier
    
    generated_node_ids = []
    for i in range(n_generated):
        new_node_id = n_original + i
        generated_node_ids.append(new_node_id)
        G_augmented.add_node(new_node_id, is_generated=True)
        
        # Connect to top-k nearest original nodes
        top_k_indices = np.argsort(similarities[i])[-k_neighbors:]
        for orig_idx in top_k_indices:
            sim_value = similarities[i, orig_idx]
            if sim_value >= threshold:
                G_augmented.add_edge(new_node_id, orig_idx, weight=sim_value,
                                   edge_type='gen_orig', is_generated=True)
    
    # STRICT generated-generated edges
    # Find mutual top-k neighbors
    mutual_top_k = {}
    for i in range(n_generated):
        # Get top-k generated neighbors for node i
        top_k_gen_i = np.argsort(gen_similarities[i])[-k_neighbors:]
        mutual_top_k[i] = set(top_k_gen_i)
    
    # Add edges only if mutual and under cap
    gg_edge_counts = {i: 0 for i in range(n_generated)}
    for i in range(n_generated):
        node_i = n_original + i
        for j in range(i + 1, n_generated):
            if gg_edge_counts[i] >= max_gg_edges_per_node:
                break
            if gg_edge_counts[j] >= max_gg_edges_per_node:
                continue
                
            # Check mutual top-k condition
            if j in mutual_top_k[i] and i in mutual_top_k[j]:
                sim_value = gen_similarities[i, j]
                if sim_value >= threshold_gg:
                    node_j = n_original + j
                    if not G_augmented.has_edge(node_i, node_j):
                        G_augmented.add_edge(node_i, node_j, weight=sim_value,
                                           edge_type='gen_gen', is_generated=True)
                        gg_edge_counts[i] += 1
                        gg_edge_counts[j] += 1
    
    return G_augmented, generated_node_ids


def insert_nodes_density_adaptive(G_original, original_features, generated_features,
                                 base_k=10, base_tau=0.5, allow_gg=True):
    """
    AGN variant: Density-adaptive insertion
    - Adapts k and tau based on graph density
    - Optionally allows generated-generated edges (with original logic)
    """
    k_adapted, tau_adapted = compute_density_adaptive_params(G_original, base_k, base_tau)
    
    if allow_gg:
        return insert_nodes_original(G_original, original_features, generated_features,
                                    k_neighbors=k_adapted, threshold=tau_adapted)
    else:
        return insert_nodes_no_gg(G_original, original_features, generated_features,
                                 k_neighbors=k_adapted, threshold=tau_adapted)


def generate_new_nodes(model, num_samples=100, device=DEVICE):
    """Generate new node features from the trained model"""
    model.eval()
    with torch.no_grad():
        z_samples = torch.randn(num_samples, model.decoder.decoder[0].in_features).to(device)
        new_features = model.decode_nodes(z_samples)
        new_features = new_features.cpu().numpy()
    return new_features


def generate_and_insert_variant(model, G_original, original_features,
                                variant='original', num_samples=100,
                                k_neighbors=10, threshold=0.5,
                                tau_gg_multiplier=1.2, max_gg_edges_per_node=3):
    """
    Generate nodes and insert using specified variant
    
    Args:
        model: Trained VGAE model
        G_original: Original NetworkX graph
        original_features: Original node features
        variant: 'original', 'no_gg', 'strict_gg', 'density_adaptive', 'density_adaptive_no_gg'
        num_samples: Number of nodes to generate
        k_neighbors: Base k value
        threshold: Base threshold value
        tau_gg_multiplier: Multiplier for generated-generated edge threshold
        max_gg_edges_per_node: Max generated-generated edges per node
    
    Returns:
        G_augmented: Graph with generated nodes
        generated_features: Generated node features
        generated_node_ids: List of generated node IDs
    """
    # Generate new nodes
    generated_features = generate_new_nodes(model, num_samples=num_samples)
    
    # Insert using specified variant
    if variant == 'original':
        G_augmented, generated_node_ids = insert_nodes_original(
            G_original, original_features, generated_features,
            k_neighbors=k_neighbors, threshold=threshold
        )
    elif variant == 'no_gg':
        G_augmented, generated_node_ids = insert_nodes_no_gg(
            G_original, original_features, generated_features,
            k_neighbors=k_neighbors, threshold=threshold
        )
    elif variant == 'strict_gg':
        G_augmented, generated_node_ids = insert_nodes_strict_gg(
            G_original, original_features, generated_features,
            k_neighbors=k_neighbors, threshold=threshold,
            tau_gg_multiplier=tau_gg_multiplier,
            max_gg_edges_per_node=max_gg_edges_per_node
        )
    elif variant == 'density_adaptive':
        G_augmented, generated_node_ids = insert_nodes_density_adaptive(
            G_original, original_features, generated_features,
            base_k=k_neighbors, base_tau=threshold, allow_gg=True
        )
    elif variant == 'density_adaptive_no_gg':
        G_augmented, generated_node_ids = insert_nodes_density_adaptive(
            G_original, original_features, generated_features,
            base_k=k_neighbors, base_tau=threshold, allow_gg=False
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return G_augmented, generated_features, generated_node_ids
