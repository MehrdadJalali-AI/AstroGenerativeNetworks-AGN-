"""
Enhanced novelty and diversity evaluation
Replaces weak 25th percentile threshold with scientifically rigorous metrics
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
import pandas as pd


def compute_enhanced_novelty(original_features, generated_features):
    """
    Compute comprehensive novelty and diversity metrics
    
    Returns:
        novelty_metrics: Dictionary with all novelty metrics
    """
    novelty_metrics = {}
    
    # Normalize features
    def normalize(feat):
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feat / norms
    
    orig_norm = normalize(original_features)
    gen_norm = normalize(generated_features)
    
    # 1. Nearest-neighbor distances
    gen_to_orig_sim = cosine_similarity(gen_norm, orig_norm)
    gen_to_orig_dist = 1 - gen_to_orig_sim  # Convert to distance
    
    nearest_orig_distances = np.min(gen_to_orig_dist, axis=1)
    mean_dist_to_orig = np.mean(gen_to_orig_dist, axis=1)
    
    novelty_metrics['nearest_neighbor'] = {
        'mean': float(np.mean(nearest_orig_distances)),
        'std': float(np.std(nearest_orig_distances)),
        'min': float(np.min(nearest_orig_distances)),
        'max': float(np.max(nearest_orig_distances)),
        'median': float(np.median(nearest_orig_distances)),
        'q25': float(np.percentile(nearest_orig_distances, 25)),
        'q75': float(np.percentile(nearest_orig_distances, 75))
    }
    
    novelty_metrics['mean_distance_to_original'] = {
        'mean': float(np.mean(mean_dist_to_orig)),
        'std': float(np.std(mean_dist_to_orig)),
        'min': float(np.min(mean_dist_to_orig)),
        'max': float(np.max(mean_dist_to_orig))
    }
    
    # 2. Generated-to-generated distances (diversity)
    gen_to_gen_sim = cosine_similarity(gen_norm, gen_norm)
    gen_to_gen_dist = 1 - gen_to_gen_sim
    gen_to_gen_dist_flat = gen_to_gen_dist[np.triu_indices_from(gen_to_gen_dist, k=1)]
    
    novelty_metrics['generated_diversity'] = {
        'mean_pairwise_distance': float(np.mean(gen_to_gen_dist_flat)),
        'std_pairwise_distance': float(np.std(gen_to_gen_dist_flat)),
        'min_pairwise_distance': float(np.min(gen_to_gen_dist_flat)),
        'max_pairwise_distance': float(np.max(gen_to_gen_dist_flat))
    }
    
    # 3. Duplicate/near-duplicate detection
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
    duplicate_rates = {}
    
    for threshold in thresholds:
        # Count generated nodes that are very close to any original node
        near_duplicates = np.sum(nearest_orig_distances < threshold)
        duplicate_rates[f'threshold_{threshold}'] = {
            'count': int(near_duplicates),
            'rate': float(near_duplicates / len(generated_features))
        }
    
    novelty_metrics['duplicate_rates'] = duplicate_rates
    
    # Also check for duplicates within generated set
    gen_self_duplicates = {}
    gen_to_gen_nearest = []
    for i in range(len(gen_norm)):
        dists = gen_to_gen_dist[i]
        dists[i] = np.inf  # Exclude self
        gen_to_gen_nearest.append(np.min(dists))
    
    gen_to_gen_nearest = np.array(gen_to_gen_nearest)
    for threshold in thresholds:
        near_duplicates = np.sum(gen_to_gen_nearest < threshold)
        gen_self_duplicates[f'threshold_{threshold}'] = {
            'count': int(near_duplicates),
            'rate': float(near_duplicates / len(generated_features))
        }
    
    novelty_metrics['generated_self_duplicates'] = gen_self_duplicates
    
    # 4. Distribution overlap metrics (Wasserstein distance per feature)
    n_features = original_features.shape[1]
    wasserstein_distances = []
    
    for feat_idx in range(n_features):
        orig_feat = original_features[:, feat_idx]
        gen_feat = generated_features[:, feat_idx]
        wd = wasserstein_distance(orig_feat, gen_feat)
        wasserstein_distances.append(wd)
    
    novelty_metrics['wasserstein_distances'] = {
        'per_feature': [float(wd) for wd in wasserstein_distances],
        'mean': float(np.mean(wasserstein_distances)),
        'std': float(np.std(wasserstein_distances)),
        'min': float(np.min(wasserstein_distances)),
        'max': float(np.max(wasserstein_distances))
    }
    
    # 5. Coverage/diversity in latent space (if we had latent representations)
    # For now, use feature space
    gen_pairwise_dist = pdist(gen_norm, metric='cosine')
    orig_pairwise_dist = pdist(orig_norm, metric='cosine')
    
    novelty_metrics['pairwise_distance_distributions'] = {
        'generated_mean': float(np.mean(gen_pairwise_dist)),
        'generated_std': float(np.std(gen_pairwise_dist)),
        'original_mean': float(np.mean(orig_pairwise_dist)),
        'original_std': float(np.std(orig_pairwise_dist)),
        'ratio': float(np.mean(gen_pairwise_dist) / np.mean(orig_pairwise_dist)) if np.mean(orig_pairwise_dist) > 0 else 0.0
    }
    
    # 6. Entropy-based diversity (simplified)
    # Use k-means clustering to estimate coverage
    try:
        from sklearn.cluster import KMeans
        n_clusters = min(10, len(generated_features) // 10)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            gen_clusters = kmeans.fit_predict(gen_norm)
            orig_clusters = kmeans.predict(orig_norm)
            
            gen_cluster_counts = np.bincount(gen_clusters)
            orig_cluster_counts = np.bincount(orig_clusters, minlength=n_clusters)
            
            # Normalize to probabilities
            gen_probs = gen_cluster_counts / np.sum(gen_cluster_counts)
            orig_probs = orig_cluster_counts / np.sum(orig_cluster_counts)
            
            # Compute entropy
            gen_entropy = -np.sum(gen_probs * np.log(gen_probs + 1e-10))
            orig_entropy = -np.sum(orig_probs * np.log(orig_probs + 1e-10))
            
            novelty_metrics['coverage_entropy'] = {
                'generated_entropy': float(gen_entropy),
                'original_entropy': float(orig_entropy),
                'entropy_ratio': float(gen_entropy / orig_entropy) if orig_entropy > 0 else 0.0
            }
        else:
            novelty_metrics['coverage_entropy'] = {
                'generated_entropy': 0.0,
                'original_entropy': 0.0,
                'entropy_ratio': 0.0
            }
    except:
        novelty_metrics['coverage_entropy'] = {
            'generated_entropy': 0.0,
            'original_entropy': 0.0,
            'entropy_ratio': 0.0
        }
    
    # 7. Legacy metric (for comparison)
    # 25th percentile threshold
    threshold_25 = np.percentile(nearest_orig_distances, 25)
    novel_count_25 = np.sum(nearest_orig_distances > threshold_25)
    novelty_metrics['legacy_threshold_25'] = {
        'threshold': float(threshold_25),
        'novel_count': int(novel_count_25),
        'novel_percentage': float(novel_count_25 / len(generated_features) * 100)
    }
    
    return novelty_metrics


def save_novelty_metrics(novelty_metrics, dataset_name, method_name, output_dir):
    """Save novelty metrics to CSV"""
    import os
    os.makedirs(os.path.join(output_dir, 'raw', 'novelty', dataset_name), exist_ok=True)
    
    # Flatten nested structure for CSV
    flat_metrics = {}
    for key, value in novelty_metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    # Store list as string representation
                    flat_metrics[f"{key}_{subkey}"] = str(subvalue)
                else:
                    flat_metrics[f"{key}_{subkey}"] = subvalue
        else:
            flat_metrics[key] = value
    
    df = pd.DataFrame([flat_metrics])
    df['dataset'] = dataset_name
    df['method'] = method_name
    
    filepath = os.path.join(output_dir, 'raw', 'novelty', dataset_name, f'{method_name}.csv')
    df.to_csv(filepath, index=False)
    
    return filepath
