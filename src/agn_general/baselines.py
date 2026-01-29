"""
Baseline methods for node insertion comparison
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch


def random_attachment(G_original, num_nodes, k_neighbors=10):
    """
    Baseline: Random attachment
    Connect new nodes to k random existing nodes
    
    Args:
        G_original: Original NetworkX graph
        num_nodes: Number of nodes to generate
        k_neighbors: Number of neighbors to connect to
    
    Returns:
        G_augmented: Graph with randomly attached nodes
        generated_node_ids: List of generated node IDs
    """
    G_augmented = G_original.copy()
    n_original = G_original.number_of_nodes()
    original_nodes = list(G_original.nodes())
    
    generated_node_ids = []
    for i in range(num_nodes):
        new_node_id = n_original + i
        generated_node_ids.append(new_node_id)
        G_augmented.add_node(new_node_id, is_generated=True)
        
        # Connect to k random existing nodes
        neighbors = np.random.choice(original_nodes, size=min(k_neighbors, len(original_nodes)), replace=False)
        for neighbor in neighbors:
            G_augmented.add_edge(new_node_id, neighbor, is_generated=True, baseline='random')
    
    return G_augmented, generated_node_ids


def preferential_attachment(G_original, num_nodes, k_neighbors=10):
    """
    Baseline: Preferential attachment
    Connect new nodes based on degree-proportional probability
    
    Args:
        G_original: Original NetworkX graph
        num_nodes: Number of nodes to generate
        k_neighbors: Number of neighbors to connect to
    
    Returns:
        G_augmented: Graph with preferentially attached nodes
        generated_node_ids: List of generated node IDs
    """
    G_augmented = G_original.copy()
    n_original = G_original.number_of_nodes()
    original_nodes = list(G_original.nodes())
    
    # Compute degree distribution for sampling
    degrees = np.array([G_original.degree(n) for n in original_nodes])
    probs = degrees / degrees.sum() if degrees.sum() > 0 else np.ones(len(original_nodes)) / len(original_nodes)
    
    generated_node_ids = []
    for i in range(num_nodes):
        new_node_id = n_original + i
        generated_node_ids.append(new_node_id)
        G_augmented.add_node(new_node_id, is_generated=True)
        
        # Sample k neighbors with probability proportional to degree
        neighbors = np.random.choice(original_nodes, size=min(k_neighbors, len(original_nodes)), 
                                     replace=False, p=probs)
        for neighbor in neighbors:
            G_augmented.add_edge(new_node_id, neighbor, is_generated=True, baseline='preferential')
        
        # Update probabilities for next iteration (dynamic preferential attachment)
        degrees[neighbors] += 1
        probs = degrees / degrees.sum()
    
    return G_augmented, generated_node_ids


def knn_feature_space(G_original, original_features, generated_features, k_neighbors=10, threshold=0.5):
    """
    Baseline: kNN in original feature space (no VGAE training)
    Connect generated nodes to top-k similar original nodes by cosine similarity
    
    Args:
        G_original: Original NetworkX graph
        original_features: Original node features (numpy array)
        generated_features: Generated node features (numpy array)
        k_neighbors: Number of nearest neighbors
        threshold: Minimum similarity threshold
    
    Returns:
        G_augmented: Graph with kNN-attached nodes
        generated_node_ids: List of generated node IDs
    """
    G_augmented = G_original.copy()
    n_original = G_original.number_of_nodes()
    n_generated = len(generated_features)
    
    # Normalize features
    def normalize_features(feat):
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feat / norms
    
    orig_norm = normalize_features(original_features)
    gen_norm = normalize_features(generated_features)
    
    # Compute cosine similarity
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
                                   is_generated=True, baseline='knn')
    
    # Connect generated nodes to each other
    gen_similarities = cosine_similarity(gen_norm, gen_norm)
    for i in range(n_generated):
        for j in range(i + 1, n_generated):
            sim_value = gen_similarities[i, j]
            if sim_value >= threshold:
                node_i = n_original + i
                node_j = n_original + j
                if not G_augmented.has_edge(node_i, node_j):
                    G_augmented.add_edge(node_i, node_j, weight=sim_value, 
                                       is_generated=True, baseline='knn')
    
    return G_augmented, generated_node_ids


def vanilla_vgae(model, G_original, original_features, num_nodes, k_neighbors=10, threshold=0.5, device=None):
    """
    Baseline: Vanilla VGAE (decoder-only edges, no similarity insertion)
    Generate nodes using VGAE but connect edges using decoder probabilities only
    
    Args:
        model: Trained VGAE model
        G_original: Original NetworkX graph
        original_features: Original node features
        num_nodes: Number of nodes to generate
        k_neighbors: Number of neighbors to consider
        threshold: Minimum edge probability threshold
        device: Device to use (if None, uses model's device or CPU)
    
    Returns:
        G_augmented: Graph with VGAE-decoder-attached nodes
        generated_features: Generated node features
        generated_node_ids: List of generated node IDs
    """
    from .generation import generate_new_nodes
    from .config import DEVICE
    
    if device is None:
        device = DEVICE
    
    G_augmented = G_original.copy()
    n_original = G_original.number_of_nodes()
    
    # Generate node features using VGAE
    generated_features = generate_new_nodes(model, num_samples=num_nodes)
    
    # Encode original nodes to get latent representations
    model.eval()
    with torch.no_grad():
        original_features_tensor = torch.FloatTensor(original_features).to(device)
        # Get edge_index for original graph
        edge_index = []
        for u, v in G_original.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])  # undirected
        if len(edge_index) == 0:
            # Empty graph, use identity
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
        else:
            edge_index = torch.LongTensor(edge_index).t().contiguous().to(device)
        
        # Encode original nodes
        mu_orig, logvar_orig = model.encode(original_features_tensor, edge_index)
        z_orig = model.reparameterize(mu_orig, logvar_orig)
        
        # Sample latent vectors for generated nodes
        latent_dim = z_orig.shape[1]
        z_gen = torch.randn(num_nodes, latent_dim).to(device)
        
        # Compute edge probabilities using decoder
        # For each generated node, compute probability with all original nodes
        edge_probs = torch.sigmoid(torch.mm(z_gen, z_orig.t())).cpu().numpy()
    
    generated_node_ids = []
    for i in range(num_nodes):
        new_node_id = n_original + i
        generated_node_ids.append(new_node_id)
        G_augmented.add_node(new_node_id, is_generated=True)
        
        # Connect to top-k original nodes by decoder probability
        top_k_indices = np.argsort(edge_probs[i])[-k_neighbors:]
        for orig_idx in top_k_indices:
            prob_value = edge_probs[i, orig_idx]
            if prob_value >= threshold:
                G_augmented.add_edge(new_node_id, orig_idx, weight=prob_value, 
                                   is_generated=True, baseline='vanilla_vgae')
    
    # Connect generated nodes to each other using decoder
    gen_edge_probs = torch.sigmoid(torch.mm(z_gen, z_gen.t())).cpu().numpy()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob_value = gen_edge_probs[i, j]
            if prob_value >= threshold:
                node_i = n_original + i
                node_j = n_original + j
                if not G_augmented.has_edge(node_i, node_j):
                    G_augmented.add_edge(node_i, node_j, weight=prob_value, 
                                       is_generated=True, baseline='vanilla_vgae')
    
    return G_augmented, generated_features, generated_node_ids
