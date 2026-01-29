"""
Generation module for creating new nodes and edges
"""

import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
from .config import DEVICE, GENERATED_DIR, NUM_GENERATED_NODES, K_NEIGHBORS, SIMILARITY_THRESHOLD

def generate_new_nodes(model, num_samples=NUM_GENERATED_NODES, device=DEVICE):
    """
    Generate new node features from the trained model
    
    Args:
        model: Trained VGAE model
        num_samples: Number of nodes to generate
        device: Device to run on
    
    Returns:
        new_features: Generated node features (numpy array)
    """
    model.eval()
    with torch.no_grad():
        # Sample from prior distribution (standard normal)
        z_samples = torch.randn(num_samples, model.decoder.decoder[0].in_features).to(device)
        
        # Decode to node features
        new_features = model.decode_nodes(z_samples)
        new_features = new_features.cpu().numpy()
    
    return new_features

def insert_nodes_to_graph(G_original, original_features, generated_features, 
                          k_neighbors=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD):
    """
    Insert generated nodes into the graph with appropriate edges
    
    Args:
        G_original: Original NetworkX graph
        original_features: Original node features (numpy array)
        generated_features: Generated node features (numpy array)
        k_neighbors: Number of nearest neighbors to connect
        threshold: Minimum similarity threshold for edges
    
    Returns:
        G_augmented: Graph with generated nodes inserted
        node_mapping: Mapping from original node IDs to new node IDs
    """
    G_augmented = G_original.copy()
    
    n_original = len(G_original.nodes())
    n_generated = len(generated_features)
    
    # Normalize features for similarity computation
    def normalize_features(feat):
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feat / norms
    
    orig_norm = normalize_features(original_features)
    gen_norm = normalize_features(generated_features)
    
    # Compute cosine similarity between generated and original nodes
    similarities = cosine_similarity(gen_norm, orig_norm)
    
    # Add generated nodes to graph
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
                G_augmented.add_edge(new_node_id, orig_idx, weight=sim_value, is_generated=True)
    
    # Optionally connect generated nodes to each other
    gen_similarities = cosine_similarity(gen_norm, gen_norm)
    for i in range(n_generated):
        for j in range(i + 1, n_generated):
            sim_value = gen_similarities[i, j]
            if sim_value >= threshold:
                node_i = n_original + i
                node_j = n_original + j
                if not G_augmented.has_edge(node_i, node_j):
                    G_augmented.add_edge(node_i, node_j, weight=sim_value, is_generated=True)
    
    return G_augmented, generated_node_ids

def save_generated_nodes(generated_features, filename="generated_nodes.csv"):
    """Save generated node features to CSV"""
    os.makedirs(GENERATED_DIR, exist_ok=True)
    filepath = os.path.join(GENERATED_DIR, filename)
    np.savetxt(filepath, generated_features, delimiter=",", fmt="%.6f")
    print(f"Generated {len(generated_features)} nodes saved to {filepath}")
    return filepath

def generate_and_insert(model, G_original, original_features, num_samples=NUM_GENERATED_NODES,
                        k_neighbors=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD):
    """
    Complete pipeline: generate nodes and insert into graph
    
    Args:
        model: Trained VGAE model
        G_original: Original NetworkX graph
        original_features: Original node features
        num_samples: Number of nodes to generate
        k_neighbors: Number of nearest neighbors
        threshold: Similarity threshold
    
    Returns:
        G_augmented: Graph with generated nodes
        generated_features: Generated node features
        generated_node_ids: List of generated node IDs
    """
    # Generate new nodes
    generated_features = generate_new_nodes(model, num_samples=num_samples)
    
    # Insert into graph
    G_augmented, generated_node_ids = insert_nodes_to_graph(
        G_original, original_features, generated_features,
        k_neighbors=k_neighbors, threshold=threshold
    )
    
    # Save generated nodes
    save_generated_nodes(generated_features)
    
    print(f"\nGeneration Summary:")
    print(f"  Original nodes: {G_original.number_of_nodes()}")
    print(f"  Generated nodes: {num_samples}")
    print(f"  Total nodes: {G_augmented.number_of_nodes()}")
    print(f"  Original edges: {G_original.number_of_edges()}")
    print(f"  Total edges: {G_augmented.number_of_edges()}")
    print(f"  New edges: {G_augmented.number_of_edges() - G_original.number_of_edges()}")
    
    return G_augmented, generated_features, generated_node_ids
