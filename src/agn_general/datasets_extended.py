"""
Extended dataset generators for AGN evaluation
Includes non-assortative SBM variants and scale-free networks
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
from .normalization import fit_two_stage_normalize, tuple_from_normalizer
from .config import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def create_node_features(G, feature_type='standard'):
    """
    Create node features based on graph structure
    
    Args:
        G: NetworkX graph
        feature_type: 'standard' or 'extended'
    
    Returns:
        features: Node features array
    """
    features = []
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        feat = [
            G.degree(node),
            nx.clustering(G, node),
            len(neighbors),
            np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,
        ]
        
        if feature_type == 'extended':
            feat.extend([
                np.std([G.degree(n) for n in neighbors]) if len(neighbors) > 1 else 0,
                len([n for n in neighbors if G.degree(n) > np.mean([G.degree(m) for m in G.nodes()])]) / max(len(neighbors), 1),
            ])
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def load_sbm_assortative(n_nodes=1200, portion=1.0):
    """
    Assortative SBM: Strong within-community connections
    Similar to existing karate dataset
    """
    n_sample = max(int(n_nodes * portion), 1000)
    
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
    
    features = create_node_features(G, feature_type='standard')
    
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index


def load_sbm_disassortative(n_nodes=1200, portion=1.0):
    """
    Disassortative SBM: Strong between-community connections
    Bipartite-like structure
    """
    n_sample = max(int(n_nodes * portion), 1000)
    
    sizes = [n_sample // 2, n_sample - n_sample // 2]
    # Low within-community, high between-community
    probs = [[0.05, 0.25],
             [0.25, 0.05]]
    
    G = nx.stochastic_block_model(sizes, probs, seed=RANDOM_SEED)
    
    features = create_node_features(G, feature_type='standard')
    
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index


def load_sbm_core_periphery(n_nodes=1200, portion=1.0):
    """
    Core-periphery SBM: Dense core, sparse periphery
    """
    n_sample = max(int(n_nodes * portion), 1000)
    
    core_size = n_sample // 4
    periphery_size = n_sample - core_size
    
    sizes = [core_size, periphery_size]
    # Dense core, sparse periphery, moderate core-periphery connections
    probs = [[0.4, 0.1],
             [0.1, 0.05]]
    
    G = nx.stochastic_block_model(sizes, probs, seed=RANDOM_SEED)
    
    features = create_node_features(G, feature_type='extended')
    
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index


def load_zachary_real(portion=1.0):
    """
    Zachary karate club (34 nodes, real social network).
    Structural features: degree, clustering, |N(v)|, mean neighbor degree (4-D).
    """
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    features = create_node_features(G, feature_type="standard")
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    return G, features, data.edge_index


def load_lesmis_real(portion=1.0):
    """
    Les Misérables character coappearance (77 nodes, 254 edges), NetworkX built-in.
    Uses extended structural features (6-D) for richer degree structure.
    """
    G = nx.les_miserables_graph()
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    features = create_node_features(G, feature_type="extended")
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    return G, features, data.edge_index


def load_scale_free_sparse(n_nodes=2000, portion=1.0, m=2):
    """
    Sparse scale-free network (Barabási-Albert with low m)
    Tests sparse-graph failure mode
    """
    n_sample = max(int(n_nodes * portion), 1000)
    
    # Low m creates sparse network
    G = nx.barabasi_albert_graph(n_sample, m, seed=RANDOM_SEED)
    
    features = create_node_features(G, feature_type='extended')
    
    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float32)
    data.edge_index = to_undirected(data.edge_index)
    
    return G, features, data.edge_index


def normalize_features(features):
    """Z-score then per-feature min--max to [0, 1]."""
    scaled, normalizer = fit_two_stage_normalize(features)
    scaler, feat_min, feat_max = tuple_from_normalizer(normalizer)
    return scaled, scaler, feat_min, feat_max


def load_dataset_extended(dataset_name, portion=1.0):
    """
    Load extended dataset by name
    
    Supported datasets:
    - sbm_assortative: Assortative SBM
    - sbm_disassortative: Disassortative/bipartite SBM
    - sbm_core_periphery: Core-periphery SBM
    - scale_free_sparse: Sparse scale-free network
    - karate, facebook, email: Original datasets (fallback to data_loader)
    """
    loaders = {
        'sbm_assortative': lambda p: load_sbm_assortative(portion=p),
        'sbm_disassortative': lambda p: load_sbm_disassortative(portion=p),
        'sbm_core_periphery': lambda p: load_sbm_core_periphery(portion=p),
        'scale_free_sparse': lambda p: load_scale_free_sparse(portion=p, m=2),
        'zachary': lambda p: load_zachary_real(portion=p),
        'lesmis': lambda p: load_lesmis_real(portion=p),
    }
    
    if dataset_name in loaders:
        G, features, edge_index = loaders[dataset_name](portion)
        features_normalized, scaler, feat_min, feat_max = normalize_features(features)
        
        print(f"Loaded {dataset_name} dataset:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.4f}")
        print(f"  Features shape: {features_normalized.shape}")
        
        return G, features_normalized, edge_index, scaler, feat_min, feat_max
    else:
        # Fallback to original data_loader
        from .data_loader import load_dataset
        return load_dataset(dataset_name, portion)
