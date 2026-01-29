"""
Main script for Generalized AGN (Astro Generative Network)
Generates new nodes and edges for any network type
"""

import torch
import numpy as np
from .config import (
    DEVICE, HIDDEN_DIM, LATENT_DIM, NUM_GCN_LAYERS,
    EPOCHS, LEARNING_RATE, NUM_GENERATED_NODES, K_NEIGHBORS, SIMILARITY_THRESHOLD,
    DATASET_PORTION
)
from .data_loader import load_dataset
from .model import GraphEncoder, NodeDecoder, VGAE
from .training import run_training
from .generation import generate_and_insert
from .evaluation import generate_evaluation_report

def main():
    """Main execution function"""
    
    # List of datasets to evaluate
    datasets = ["karate", "facebook", "email"]
    
    print("="*80)
    print("GENERALIZED ASTRO GENERATIVE NETWORK (AGN)")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"Dataset portion: {DATASET_PORTION*100}%")
    print(f"Number of generated nodes: {NUM_GENERATED_NODES}")
    print("="*80 + "\n")
    
    results_summary = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Load dataset
            G, features, edge_index, scaler, feat_min, feat_max = load_dataset(
                dataset_name=dataset_name,
                portion=DATASET_PORTION
            )
            
            # Convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
            edge_index_tensor = edge_index.to(DEVICE)
            
            # Initialize model
            input_dim = features.shape[1]
            encoder = GraphEncoder(input_dim, HIDDEN_DIM, LATENT_DIM, NUM_GCN_LAYERS)
            decoder = NodeDecoder(LATENT_DIM, HIDDEN_DIM, input_dim)
            model = VGAE(encoder, decoder).to(DEVICE)
            
            print(f"\nModel Architecture:")
            print(f"  Input features: {input_dim}")
            print(f"  Hidden dimension: {HIDDEN_DIM}")
            print(f"  Latent dimension: {LATENT_DIM}")
            print(f"  GCN layers: {NUM_GCN_LAYERS}")
            
            # Train model
            print(f"\nTraining model...")
            losses = run_training(
                model, features_tensor, edge_index_tensor,
                epochs=EPOCHS, lr=LEARNING_RATE
            )
            
            # Generate new nodes and insert into graph
            print(f"\nGenerating new nodes...")
            G_augmented, generated_features, generated_node_ids = generate_and_insert(
                model, G, features, num_samples=NUM_GENERATED_NODES,
                k_neighbors=K_NEIGHBORS, threshold=SIMILARITY_THRESHOLD
            )
            
            # Evaluate
            print(f"\nEvaluating results...")
            eval_results = generate_evaluation_report(
                G, G_augmented, features, generated_features,
                dataset_name=dataset_name
            )
            
            # Run comprehensive evaluation with baselines and task-level metrics
            print(f"\nRunning comprehensive evaluation with baselines...")
            try:
                from .comprehensive_evaluation import run_comprehensive_evaluation
                comprehensive_results = run_comprehensive_evaluation(
                    model, G, features, dataset_name,
                    num_generated=NUM_GENERATED_NODES,
                    k=K_NEIGHBORS,
                    threshold=SIMILARITY_THRESHOLD
                )
                print(f"✓ Comprehensive evaluation completed for {dataset_name}")
            except Exception as e:
                print(f"⚠ Comprehensive evaluation failed for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
            
            # Run ablation study
            print(f"\nRunning ablation study...")
            try:
                from .ablation_analysis import run_ablation_study
                ablation_results = run_ablation_study(
                    model, G, features, dataset_name,
                    num_generated=NUM_GENERATED_NODES
                )
                print(f"✓ Ablation study completed for {dataset_name}")
            except Exception as e:
                print(f"⚠ Ablation study failed for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
            
            # Get novelty metrics (handle both old and new format)
            novelty_data = eval_results.get('novelty', {})
            novelty_score = novelty_data.get('avg_novelty_score', novelty_data.get('avg_novelty', 0.0))
            novelty_percentage = novelty_data.get('novelty_percentage_threshold', novelty_data.get('novelty_percentage', 0.0))
            
            results_summary[dataset_name] = {
                'original_nodes': G.number_of_nodes(),
                'generated_nodes': NUM_GENERATED_NODES,
                'total_nodes': G_augmented.number_of_nodes(),
                'original_edges': G.number_of_edges(),
                'total_edges': G_augmented.number_of_edges(),
                'novelty_score': novelty_score,
                'novelty_percentage': novelty_percentage
            }
            
            print(f"✓ Completed {dataset_name}")
            
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for dataset_name, results in results_summary.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Original nodes: {results['original_nodes']}")
        print(f"  Generated nodes: {results['generated_nodes']}")
        print(f"  Total nodes: {results['total_nodes']}")
        print(f"  Original edges: {results['original_edges']}")
        print(f"  Total edges: {results['total_edges']}")
        print(f"  Novelty score: {results['novelty_score']:.4f}")
        print(f"  Novelty percentage: {results['novelty_percentage']:.1f}%")
    print("="*80)
    print("\nAll results saved to results/ directory")
    print("="*80)

if __name__ == "__main__":
    main()
