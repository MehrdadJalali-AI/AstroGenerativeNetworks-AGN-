"""
Comprehensive evaluation script for AGN with baselines and task-level evaluation
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import json
from .config import RESULTS_DIR, PLOTS_DIR, NUM_GENERATED_NODES, K_NEIGHBORS, SIMILARITY_THRESHOLD, RANDOM_SEED
from .evaluation import compute_topology_metrics, novelty_analysis, save_metrics_to_csv
from .baselines import random_attachment, preferential_attachment, knn_feature_space, vanilla_vgae
from .task_evaluation import evaluate_link_prediction, evaluate_node_classification, evaluate_community_stability, evaluate_robustness_missing_edges
from .generation import generate_and_insert, generate_new_nodes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_comprehensive_evaluation(model, G_original, original_features, dataset_name, 
                                 num_generated=NUM_GENERATED_NODES, k=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD,
                                 seed=RANDOM_SEED):
    """
    Run comprehensive evaluation including baselines and task-level metrics
    
    Args:
        model: Trained VGAE model
        G_original: Original NetworkX graph
        original_features: Original node features
        dataset_name: Name of dataset
        num_generated: Number of nodes to generate
        k: Number of neighbors
        threshold: Similarity threshold
        seed: Random seed
    
    Returns:
        results: Dictionary with all evaluation results
    """
    np.random.seed(seed)
    
    results = {
        'dataset': dataset_name,
        'num_generated': num_generated,
        'k': k,
        'threshold': threshold,
        'methods': {}
    }
    
    # 1. AGN (our method)
    print(f"\n{'='*60}")
    print(f"Evaluating AGN on {dataset_name}")
    print(f"{'='*60}")
    
    G_agn, features_agn, node_ids_agn = generate_and_insert(
        model, G_original, original_features, 
        num_samples=num_generated, k_neighbors=k, threshold=threshold
    )
    
    metrics_agn_before = compute_topology_metrics(G_original)
    metrics_agn_after = compute_topology_metrics(G_agn)
    novelty_agn = novelty_analysis(original_features, features_agn)
    
    # Task-level evaluation
    link_pred_agn = evaluate_link_prediction(G_agn, seed=seed)
    # For node classification, use original features only (augmented graph has more nodes)
    n_original = G_original.number_of_nodes()
    node_class_agn = evaluate_node_classification(G_agn, original_features, seed=seed, original_node_count=n_original)
    comm_stab_agn = evaluate_community_stability(G_original, G_agn, seed=seed)
    robustness_agn = evaluate_robustness_missing_edges(G_original, G_agn, seed=seed)
    
    results['methods']['AGN'] = {
        'topology_before': metrics_agn_before,
        'topology_after': metrics_agn_after,
        'novelty': novelty_agn,
        'link_prediction': link_pred_agn,
        'node_classification': node_class_agn,
        'community_stability': comm_stab_agn,
        'robustness': robustness_agn
    }
    
    # 2. Random Attachment Baseline
    print(f"\nEvaluating Random Attachment baseline...")
    G_random, node_ids_random = random_attachment(G_original, num_generated, k_neighbors=k)
    metrics_random_after = compute_topology_metrics(G_random)
    link_pred_random = evaluate_link_prediction(G_random, seed=seed)
    node_class_random = evaluate_node_classification(G_random, original_features, seed=seed, original_node_count=n_original)
    comm_stab_random = evaluate_community_stability(G_original, G_random, seed=seed)
    
    results['methods']['Random'] = {
        'topology_before': metrics_agn_before,
        'topology_after': metrics_random_after,
        'link_prediction': link_pred_random,
        'node_classification': node_class_random,
        'community_stability': comm_stab_random
    }
    
    # 3. Preferential Attachment Baseline
    print(f"\nEvaluating Preferential Attachment baseline...")
    G_pref, node_ids_pref = preferential_attachment(G_original, num_generated, k_neighbors=k)
    metrics_pref_after = compute_topology_metrics(G_pref)
    link_pred_pref = evaluate_link_prediction(G_pref, seed=seed)
    node_class_pref = evaluate_node_classification(G_pref, original_features, seed=seed, original_node_count=n_original)
    comm_stab_pref = evaluate_community_stability(G_original, G_pref, seed=seed)
    
    results['methods']['Preferential'] = {
        'topology_before': metrics_agn_before,
        'topology_after': metrics_pref_after,
        'link_prediction': link_pred_pref,
        'node_classification': node_class_pref,
        'community_stability': comm_stab_pref
    }
    
    # 4. kNN Feature Space Baseline
    print(f"\nEvaluating kNN Feature Space baseline...")
    # Generate random features for kNN baseline (to match number of generated nodes)
    np.random.seed(seed)
    features_knn = np.random.rand(num_generated, original_features.shape[1])
    # Normalize to match feature distribution
    features_knn = (features_knn - features_knn.min(axis=0)) / (features_knn.max(axis=0) - features_knn.min(axis=0) + 1e-10)
    features_knn = features_knn * (original_features.max(axis=0) - original_features.min(axis=0)) + original_features.min(axis=0)
    
    G_knn, node_ids_knn = knn_feature_space(G_original, original_features, features_knn, 
                                           k_neighbors=k, threshold=threshold)
    metrics_knn_after = compute_topology_metrics(G_knn)
    novelty_knn = novelty_analysis(original_features, features_knn)
    link_pred_knn = evaluate_link_prediction(G_knn, seed=seed)
    node_class_knn = evaluate_node_classification(G_knn, original_features, seed=seed, original_node_count=n_original)
    comm_stab_knn = evaluate_community_stability(G_original, G_knn, seed=seed)
    
    results['methods']['kNN'] = {
        'topology_before': metrics_agn_before,
        'topology_after': metrics_knn_after,
        'novelty': novelty_knn,
        'link_prediction': link_pred_knn,
        'node_classification': node_class_knn,
        'community_stability': comm_stab_knn
    }
    
    # 5. Vanilla VGAE Baseline
    print(f"\nEvaluating Vanilla VGAE baseline...")
    try:
        from .config import DEVICE
        G_vgae, features_vgae, node_ids_vgae = vanilla_vgae(
            model, G_original, original_features, num_generated, k_neighbors=k, threshold=threshold, device=DEVICE
        )
        metrics_vgae_after = compute_topology_metrics(G_vgae)
        novelty_vgae = novelty_analysis(original_features, features_vgae)
        link_pred_vgae = evaluate_link_prediction(G_vgae, seed=seed)
        node_class_vgae = evaluate_node_classification(G_vgae, original_features, seed=seed, original_node_count=n_original)
        comm_stab_vgae = evaluate_community_stability(G_original, G_vgae, seed=seed)
        
        results['methods']['VanillaVGAE'] = {
            'topology_before': metrics_agn_before,
            'topology_after': metrics_vgae_after,
            'novelty': novelty_vgae,
            'link_prediction': link_pred_vgae,
            'node_classification': node_class_vgae,
            'community_stability': comm_stab_vgae
        }
    except Exception as e:
        print(f"Vanilla VGAE baseline failed: {e}")
        results['methods']['VanillaVGAE'] = None
    
    # Save results
    save_comprehensive_results(results, dataset_name)
    
    return results


def save_comprehensive_results(results, dataset_name):
    """Save comprehensive evaluation results to CSV and JSON"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create comparison tables
    methods = [m for m in results['methods'].keys() if results['methods'][m] is not None]
    
    # Topology comparison
    topology_data = []
    for method in methods:
        m_data = results['methods'][method]
        if 'topology_after' in m_data:
            metrics = m_data['topology_after']
            row = {'Method': method}
            for key in ['num_nodes', 'num_edges', 'density', 'avg_degree', 'avg_clustering', 
                       'modularity', 'avg_shortest_path_length', 'assortativity']:
                if key in metrics:
                    row[key] = metrics[key]
            topology_data.append(row)
    
    df_topology = pd.DataFrame(topology_data)
    df_topology.to_csv(os.path.join(RESULTS_DIR, f'{dataset_name}_baseline_topology_comparison.csv'), index=False)
    
    # Task-level comparison
    task_data = []
    for method in methods:
        m_data = results['methods'][method]
        row = {'Method': method}
        
        if 'link_prediction' in m_data:
            row['LinkPred_AUC'] = m_data['link_prediction'].get('auc', np.nan)
            row['LinkPred_AP'] = m_data['link_prediction'].get('ap', np.nan)
        
        if 'node_classification' in m_data:
            row['NodeClass_Accuracy'] = m_data['node_classification'].get('accuracy', np.nan)
            row['NodeClass_F1'] = m_data['node_classification'].get('f1', np.nan)
        
        if 'community_stability' in m_data:
            row['Community_NMI'] = m_data['community_stability'].get('nmi', np.nan)
            row['Community_ARI'] = m_data['community_stability'].get('ari', np.nan)
        
        task_data.append(row)
    
    df_tasks = pd.DataFrame(task_data)
    df_tasks.to_csv(os.path.join(RESULTS_DIR, f'{dataset_name}_baseline_task_comparison.csv'), index=False)
    
    # Novelty comparison
    novelty_data = []
    for method in methods:
        m_data = results['methods'][method]
        if 'novelty' in m_data:
            nov = m_data['novelty']
            row = {'Method': method}
            row['MinDist_Mean'] = nov.get('min_distance_mean', np.nan)
            row['MinDist_Std'] = nov.get('min_distance_std', np.nan)
            row['MeanDist_Mean'] = nov.get('mean_distance_mean', np.nan)
            row['DuplicationRate'] = nov.get('duplication_rate', np.nan)
            row['NoveltyPct_Threshold'] = nov.get('novelty_percentage_threshold', np.nan)
            novelty_data.append(row)
    
    if novelty_data:
        df_novelty = pd.DataFrame(novelty_data)
        df_novelty.to_csv(os.path.join(RESULTS_DIR, f'{dataset_name}_baseline_novelty_comparison.csv'), index=False)
    
    # Save full results as JSON
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    with open(os.path.join(RESULTS_DIR, f'{dataset_name}_comprehensive_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"  - {dataset_name}_baseline_topology_comparison.csv")
    print(f"  - {dataset_name}_baseline_task_comparison.csv")
    if novelty_data:
        print(f"  - {dataset_name}_baseline_novelty_comparison.csv")
    print(f"  - {dataset_name}_comprehensive_results.json")
