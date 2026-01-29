"""
Data loader for general networks (social networks, etc.)
Supports multiple network datasets
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.preprocessing import StandardScaler
import os
from .config import DATA_DIR, DATASET_PORTION, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def load_karate_club(portion=1.0):
    """
    Load Karate Club-like network with 1000+ nodes
    Creates a network with community structure similar to Karate Club
    Returns: NetworkX graph, node features, edge_index
    """
    # Create a larger network with community structure
    n_nodes = 1200
    n_sample = max(int(n_nodes * portion), 1000)  # At least 1000 nodes
    
    # Create a graph with strong community structure (like Karate Club)
    # Use stochastic block model for community structure
    sizes = [n_sample // 3, n_sample // 3, n_sample - 2 * (n_sample // 3)]
    probs = [[0.3, 0.05, 0.05],
             [0.05, 0.3, 0.05],
             [0.05, 0.05, 0.3]]
    
    G = nx.stochastic_block_model(sizes, probs, seed=RANDOM_SEED)
    
    # Add some random edges for connectivity
    for _ in range(n_sample // 20):
        u = np.random.randint(0, n_sample)
        v = np.random.randint(0, n_sample)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    
    # Create node features (degree, clustering coefficient, etc.)
    features = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        feat = [
            G.degree(node),
            nx.clustering(G, node),
            len(neighbors),
            np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,  # Avg neighbor degree
        ]
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    # Convert to PyTorch Geometric format
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index

def load_facebook_ego(portion=1.0):
    """
    Load Facebook ego-like network with 1000+ nodes
    Creates a synthetic social network with community structure
    """
    # Create a larger synthetic social network with community structure
    n_nodes = 1500
    n_sample = max(int(n_nodes * portion), 1000)  # At least 1000 nodes
    
    # Create multiple communities (like Facebook friend groups)
    n_communities = 5
    community_size = n_sample // n_communities
    
    G = nx.Graph()
    G.add_nodes_from(range(n_sample))
    
    # Add edges within communities (high probability)
    for comm_id in range(n_communities):
        start = comm_id * community_size
        end = start + community_size if comm_id < n_communities - 1 else n_sample
        comm_nodes = list(range(start, end))
        
        for i in comm_nodes:
            for j in comm_nodes:
                if i != j and np.random.random() < 0.25:
                    G.add_edge(i, j)
    
    # Add edges between communities (lower probability)
    for _ in range(n_sample // 10):
        u = np.random.randint(0, n_sample)
        v = np.random.randint(0, n_sample)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    
    # Create node features
    features = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        feat = [
            G.degree(node),
            nx.clustering(G, node),
            len(neighbors),
            np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,
            len([n for n in neighbors if G.degree(n) > np.mean([G.degree(m) for m in G.nodes()])]) / max(len(neighbors), 1),  # Fraction of high-degree neighbors
        ]
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index

def load_email_network(portion=1.0):
    """
    Load email-like network with 1000+ nodes
    Creates a scale-free network (Barabási-Albert) simulating email communication
    """
    # Create a larger scale-free network (Barabási-Albert)
    n_nodes = 2000
    n_sample = max(int(n_nodes * portion), 1000)  # At least 1000 nodes
    
    # Barabási-Albert: m edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(n_sample, 4, seed=RANDOM_SEED)
    
    # Create node features
    features = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        feat = [
            G.degree(node),
            nx.clustering(G, node),
            len(neighbors),
            np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,
            np.std([G.degree(n) for n in neighbors]) if len(neighbors) > 1 else 0,  # Std of neighbor degrees
            len([n for n in neighbors if G.degree(n) > G.degree(node)]) / max(len(neighbors), 1),  # Fraction of higher-degree neighbors
        ]
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index

def normalize_features(features):
    """Normalize features to [0, 1] range"""
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    # Scale to [0, 1]
    features_min = features_normalized.min(axis=0)
    features_max = features_normalized.max(axis=0)
    feature_range = features_max - features_min
    feature_range[feature_range == 0] = 1.0
    features_scaled = (features_normalized - features_min) / feature_range
    return features_scaled, scaler, features_min, features_max

def load_dataset(dataset_name="karate", portion=DATASET_PORTION):
    """
    Load a dataset by name
    
    Args:
        dataset_name: Name of dataset ("karate", "facebook", "email")
        portion: Fraction of dataset to use (0.0 to 1.0)
    
    Returns:
        G: NetworkX graph
        features: Node features (numpy array)
        edge_index: Edge indices (torch tensor)
        feature_scaler: Scaler for denormalization
        feature_min: Min values for denormalization
        feature_max: Max values for denormalization
    """
    loaders = {
        "karate": load_karate_club,
        "facebook": load_facebook_ego,
        "email": load_email_network,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    G, features, edge_index = loaders[dataset_name](portion=portion)
    
    # Normalize features
    features_normalized, scaler, feat_min, feat_max = normalize_features(features)
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Features shape: {features_normalized.shape}")
    
    return G, features_normalized, edge_index, scaler, feat_min, feat_max
