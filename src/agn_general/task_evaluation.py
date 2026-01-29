"""
Task-level evaluation for augmented graphs
Includes link prediction, node classification, and community stability
"""

import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.data import Data
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def evaluate_link_prediction(G, test_ratio=0.1, val_ratio=0.05, seed=42):
    """
    Evaluate link prediction performance on a graph
    
    Args:
        G: NetworkX graph
        test_ratio: Ratio of edges for testing
        val_ratio: Ratio of edges for validation
        seed: Random seed
    
    Returns:
        results: Dictionary with AUC and AP scores
    """
    np.random.seed(seed)
    
    # Convert to edge list
    edges = list(G.edges())
    if len(edges) == 0:
        return {'auc': 0.0, 'ap': 0.0}
    
    # Split edges
    edges = np.array(edges)
    n_edges = len(edges)
    
    # Create train/val/test split
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)
    n_train = n_edges - n_test - n_val
    
    indices = np.random.permutation(n_edges)
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test+n_val]
    train_indices = indices[n_test+n_val:]
    
    train_edges = edges[train_indices]
    val_edges = edges[val_indices]
    test_edges = edges[test_indices]
    
    # Create training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    
    # Generate negative samples (non-edges)
    all_nodes = list(G.nodes())
    n_neg_test = len(test_edges)
    neg_test_edges = []
    while len(neg_test_edges) < n_neg_test:
        u, v = np.random.choice(all_nodes, size=2, replace=False)
        if not G_train.has_edge(u, v) and u != v:
            neg_test_edges.append([u, v])
    
    # Simple heuristic: use common neighbors as score
    def link_score(u, v):
        neighbors_u = set(G_train.neighbors(u)) if u in G_train else set()
        neighbors_v = set(G_train.neighbors(v)) if v in G_train else set()
        common = len(neighbors_u & neighbors_v)
        total = len(neighbors_u | neighbors_v)
        return common / (total + 1e-10)
    
    # Score test edges
    pos_scores = [link_score(u, v) for u, v in test_edges]
    neg_scores = [link_score(u, v) for u, v in neg_test_edges]
    
    # Compute metrics
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except:
        auc = 0.0
        ap = 0.0
    
    return {'auc': auc, 'ap': ap}


def evaluate_node_classification(G, features, labels=None, test_ratio=0.2, seed=42, original_node_count=None):
    """
    Evaluate node classification performance
    
    Args:
        G: NetworkX graph
        features: Node features (numpy array, shape: [num_nodes, feature_dim])
                 If original_node_count is provided, only uses features for original nodes
        labels: Node labels (numpy array, shape: [num_nodes])
                If None, uses community detection as pseudo-labels
        test_ratio: Ratio of nodes for testing
        seed: Random seed
        original_node_count: Number of original nodes (if None, assumes features match all nodes)
    
    Returns:
        results: Dictionary with accuracy and F1 scores
    """
    np.random.seed(seed)
    
    # Handle case where graph has more nodes than features (augmented graph)
    nodes = list(G.nodes())
    if original_node_count is not None and len(features) < G.number_of_nodes():
        # Only use original nodes for classification
        original_nodes = [n for n in nodes if n < original_node_count]
        node_to_idx = {n: i for i, n in enumerate(original_nodes)}
        features_subset = features[:original_node_count]
        G_subset = G.subgraph(original_nodes).copy()
    else:
        original_nodes = nodes
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        features_subset = features
        G_subset = G
    
    # If no labels provided, use community detection
    if labels is None:
        try:
            communities = nx.community.louvain_communities(G_subset, seed=seed)
            labels = np.zeros(len(original_nodes))
            for i, comm in enumerate(communities):
                for node in comm:
                    if node in node_to_idx:
                        labels[node_to_idx[node]] = i
        except:
            # Fallback: use degree-based labels
            degrees = np.array([G_subset.degree(n) for n in original_nodes])
            labels = (degrees > np.median(degrees)).astype(int)
    
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Split nodes
    node_indices = np.arange(len(original_nodes))
    train_idx, test_idx = train_test_split(node_indices, test_size=test_ratio, 
                                          random_state=seed, stratify=labels)
    
    # Extract features and labels
    X_train = features_subset[train_idx]
    y_train = labels[train_idx]
    X_test = features_subset[test_idx]
    y_test = labels[test_idx]
    
    # Train simple classifier (Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # F1 score (macro average)
    try:
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    except:
        f1 = 0.0
    
    return {'accuracy': accuracy, 'f1': f1}


def evaluate_community_stability(G_original, G_augmented, original_nodes=None, seed=42):
    """
    Evaluate community stability after augmentation
    
    Args:
        G_original: Original graph
        G_augmented: Augmented graph
        original_nodes: List of original node IDs (if None, assumes first N nodes)
        seed: Random seed
    
    Returns:
        results: Dictionary with NMI and ARI scores
    """
    if original_nodes is None:
        n_orig = G_original.number_of_nodes()
        original_nodes = list(range(n_orig))
    
    # Detect communities in original graph
    try:
        comm_orig = nx.community.louvain_communities(G_original, seed=seed)
    except:
        comm_orig = [set(G_original.nodes())]  # Fallback: single community
    
    # Create label vector for original nodes
    labels_orig = np.zeros(len(original_nodes))
    for i, comm in enumerate(comm_orig):
        for node in comm:
            if node in original_nodes:
                idx = original_nodes.index(node)
                labels_orig[idx] = i
    
    # Detect communities in augmented graph
    try:
        comm_aug = nx.community.louvain_communities(G_augmented, seed=seed)
    except:
        comm_aug = [set(G_augmented.nodes())]
    
    # Create label vector for original nodes in augmented graph
    labels_aug = np.zeros(len(original_nodes))
    for i, comm in enumerate(comm_aug):
        for node in comm:
            if node in original_nodes:
                idx = original_nodes.index(node)
                labels_aug[idx] = i
    
    # Compute NMI and ARI
    try:
        nmi = normalized_mutual_info_score(labels_orig, labels_aug)
    except:
        nmi = 0.0
    
    try:
        ari = adjusted_rand_score(labels_orig, labels_aug)
    except:
        ari = 0.0
    
    return {'nmi': nmi, 'ari': ari}


def evaluate_robustness_missing_edges(G_original, G_augmented, missing_ratio=0.1, seed=42):
    """
    Evaluate robustness: remove p% edges, augment, measure recovery
    
    Args:
        G_original: Original graph
        G_augmented: Augmented graph
        missing_ratio: Ratio of edges to remove
        seed: Random seed
    
    Returns:
        results: Dictionary with recovery metrics
    """
    np.random.seed(seed)
    
    # Remove edges from original
    edges_to_remove = int(G_original.number_of_edges() * missing_ratio)
    edges = list(G_original.edges())
    remove_indices = np.random.choice(len(edges), size=edges_to_remove, replace=False)
    edges_removed = [edges[i] for i in remove_indices]
    
    G_degraded = G_original.copy()
    G_degraded.remove_edges_from(edges_removed)
    
    # Evaluate link prediction on degraded graph
    degraded_results = evaluate_link_prediction(G_degraded, seed=seed)
    
    # Evaluate link prediction on augmented graph
    augmented_results = evaluate_link_prediction(G_augmented, seed=seed)
    
    # Recovery: improvement in link prediction
    recovery_auc = augmented_results['auc'] - degraded_results['auc']
    recovery_ap = augmented_results['ap'] - degraded_results['ap']
    
    return {
        'degraded_auc': degraded_results['auc'],
        'degraded_ap': degraded_results['ap'],
        'augmented_auc': augmented_results['auc'],
        'augmented_ap': augmented_results['ap'],
        'recovery_auc': recovery_auc,
        'recovery_ap': recovery_ap
    }
