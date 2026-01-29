"""
Ablation and sensitivity analysis for AGN
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
from .config import RESULTS_DIR, NUM_GENERATED_NODES, K_NEIGHBORS, SIMILARITY_THRESHOLD, RANDOM_SEED
from .evaluation import compute_topology_metrics, novelty_analysis
from .generation import generate_and_insert, generate_new_nodes, insert_nodes_to_graph


def ablation_without_similarity(model, G_original, original_features, num_generated=NUM_GENERATED_NODES):
    """
    Ablation: Generate nodes but connect using decoder probabilities only (no similarity)
    """
    from .baselines import vanilla_vgae
    from .config import DEVICE
    
    G_abl, features_abl, _ = vanilla_vgae(
        model, G_original, original_features, num_generated, 
        k_neighbors=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD, device=DEVICE
    )
    
    metrics = compute_topology_metrics(G_abl)
    novelty = novelty_analysis(original_features, features_abl)
    
    return {
        'graph': G_abl,
        'features': features_abl,
        'topology': metrics,
        'novelty': novelty,
        'method': 'without_similarity'
    }


def ablation_without_decoder(G_original, original_features, num_generated=NUM_GENERATED_NODES, 
                             k=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD):
    """
    Ablation: Pure similarity kNN insertion (no VGAE decoder, use random features)
    """
    from .baselines import knn_feature_space
    
    # Generate random features
    np.random.seed(RANDOM_SEED)
    random_features = np.random.rand(num_generated, original_features.shape[1])
    # Normalize to match feature distribution
    random_features = (random_features - random_features.min(axis=0)) / (random_features.max(axis=0) - random_features.min(axis=0) + 1e-10)
    random_features = random_features * (original_features.max(axis=0) - original_features.min(axis=0)) + original_features.min(axis=0)
    
    G_abl, _ = knn_feature_space(G_original, original_features, random_features, 
                                 k_neighbors=k, threshold=threshold)
    
    metrics = compute_topology_metrics(G_abl)
    novelty = novelty_analysis(original_features, random_features)
    
    return {
        'graph': G_abl,
        'features': random_features,
        'topology': metrics,
        'novelty': novelty,
        'method': 'without_decoder'
    }


def sensitivity_analysis(model, G_original, original_features, num_generated=NUM_GENERATED_NODES):
    """
    Sensitivity analysis: vary k and threshold parameters
    
    Returns:
        results: Dictionary with results for each parameter combination
    """
    k_values = [5, 10, 20]
    threshold_values = [0.3, 0.5, 0.7]
    
    results = []
    
    for k in k_values:
        for threshold in threshold_values:
            print(f"\nSensitivity: k={k}, threshold={threshold}")
            
            try:
                G_aug, features_aug, _ = generate_and_insert(
                    model, G_original, original_features,
                    num_samples=num_generated, k_neighbors=k, threshold=threshold
                )
                
                metrics = compute_topology_metrics(G_aug)
                novelty = novelty_analysis(original_features, features_aug)
                
                results.append({
                    'k': k,
                    'threshold': threshold,
                    'density': metrics.get('density', np.nan),
                    'avg_clustering': metrics.get('avg_clustering', np.nan),
                    'modularity': metrics.get('modularity', np.nan),
                    'min_dist_mean': novelty.get('min_distance_mean', np.nan),
                    'mean_dist_mean': novelty.get('mean_distance_mean', np.nan),
                    'duplication_rate': novelty.get('duplication_rate', np.nan),
                    'num_edges': metrics.get('num_edges', np.nan),
                })
            except Exception as e:
                print(f"Failed for k={k}, threshold={threshold}: {e}")
                results.append({
                    'k': k,
                    'threshold': threshold,
                    'density': np.nan,
                    'avg_clustering': np.nan,
                    'modularity': np.nan,
                    'min_dist_mean': np.nan,
                    'mean_dist_mean': np.nan,
                    'duplication_rate': np.nan,
                    'num_edges': np.nan,
                })
    
    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_analysis.csv'), index=False)
    
    return df


def run_ablation_study(model, G_original, original_features, dataset_name, num_generated=NUM_GENERATED_NODES):
    """
    Run complete ablation study
    
    Returns:
        results: Dictionary with ablation results
    """
    results = {
        'dataset': dataset_name,
        'ablations': {},
        'sensitivity': None
    }
    
    # Ablation 1: Without similarity insertion
    print("\n" + "="*60)
    print("Ablation: Without Similarity Insertion")
    print("="*60)
    try:
        abl1 = ablation_without_similarity(model, G_original, original_features, num_generated)
        results['ablations']['without_similarity'] = abl1
    except Exception as e:
        print(f"Ablation 1 failed: {e}")
        results['ablations']['without_similarity'] = None
    
    # Ablation 2: Without decoder (pure kNN)
    print("\n" + "="*60)
    print("Ablation: Without Decoder (Pure kNN)")
    print("="*60)
    try:
        abl2 = ablation_without_decoder(G_original, original_features, num_generated)
        results['ablations']['without_decoder'] = abl2
    except Exception as e:
        print(f"Ablation 2 failed: {e}")
        results['ablations']['without_decoder'] = None
    
    # Sensitivity analysis
    print("\n" + "="*60)
    print("Sensitivity Analysis: Varying k and threshold")
    print("="*60)
    try:
        sensitivity_df = sensitivity_analysis(model, G_original, original_features, num_generated)
        results['sensitivity'] = sensitivity_df.to_dict('records')
    except Exception as e:
        print(f"Sensitivity analysis failed: {e}")
        results['sensitivity'] = None
    
    return results
