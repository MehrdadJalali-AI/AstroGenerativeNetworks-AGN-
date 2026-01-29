# Project Structure

## Overview

This is the **Astro Generative Network (AGN)** - a general framework for generating new nodes and edges in any network type using Variational Graph Autoencoders (VGAE).

## Directory Structure

```
AGN-General/
├── src/
│   └── agn_general/              # Main AGN module
│       ├── __init__.py
│       ├── config.py             # Configuration parameters
│       ├── data_loader.py        # Data loaders for networks
│       ├── model.py              # VGAE model architecture
│       ├── training.py           # Training loop
│       ├── generation.py         # Node generation and insertion
│       ├── evaluation.py         # Evaluation metrics
│       └── main.py               # Main execution script
│
├── data/                         # Input data directory (empty, ready for datasets)
├── results/                      # Output directory
│   ├── models/                  # Trained model checkpoints
│   ├── generated/               # Generated node features
│   └── plots/                   # Visualization plots
│
├── run_agn.py                    # Entry point script
├── requirements.txt             # Python dependencies
│
└── Documentation/
    ├── README.md                 # Main documentation
    ├── QUICKSTART.md            # Quick start guide
    ├── SUMMARY.md               # Implementation summary
    ├── CLEANUP_SUMMARY.md       # Cleanup summary
    └── PROJECT_STRUCTURE.md     # This file
```

## Active Code Files

### Core Module (`src/agn_general/`)

1. **`config.py`**: Configuration parameters
   - Model hyperparameters (hidden dim, latent dim, etc.)
   - Training parameters (epochs, learning rate, etc.)
   - Generation parameters (number of nodes, neighbors, threshold)
   - Directory paths

2. **`data_loader.py`**: Data loading functions
   - `load_karate_club()`: Zachary's Karate Club network
   - `load_facebook_ego()`: Synthetic social network
   - `load_email_network()`: Scale-free network
   - `load_dataset()`: Main loader function
   - `normalize_features()`: Feature normalization

3. **`model.py`**: Model architecture
   - `GraphEncoder`: GCN-based encoder
   - `NodeDecoder`: Feature decoder
   - `VGAE`: Variational Graph Autoencoder

4. **`training.py`**: Training module
   - `loss_function()`: VGAE loss (reconstruction + KL divergence)
   - `train_epoch()`: Single epoch training
   - `run_training()`: Main training loop with early stopping

5. **`generation.py`**: Generation module
   - `generate_new_nodes()`: Generate node features from latent space
   - `insert_nodes_to_graph()`: Insert nodes with similarity-based edges
   - `generate_and_insert()`: Complete generation pipeline

6. **`evaluation.py`**: Evaluation module
   - `compute_topology_metrics()`: Network topology metrics
   - `novelty_analysis()`: Novelty analysis
   - `plot_comparison()`: Visualization plots
   - `generate_evaluation_report()`: Comprehensive evaluation

7. **`main.py`**: Main execution script
   - Loads multiple datasets
   - Trains models
   - Generates nodes
   - Evaluates results

## Usage

### Run AGN

```bash
python3 run_agn.py
```

This will:
1. Load three network datasets (Karate Club, Facebook Ego, Email Network)
2. Train VGAE models on each dataset
3. Generate new nodes for each network
4. Insert nodes with appropriate edges
5. Evaluate and visualize results

### Configuration

Edit `src/agn_general/config.py` to customize:
- Model architecture (hidden dimensions, layers)
- Training parameters (epochs, learning rate)
- Generation parameters (number of nodes, neighbors)
- Dataset portion (fraction of dataset to use)

### Add Custom Datasets

Modify `src/agn_general/data_loader.py` to add your own network datasets:

```python
def load_your_dataset(portion=1.0):
    # Load your network
    G = nx.read_edgelist("your_network.edgelist")
    
    # Create node features
    features = []
    for node in G.nodes():
        feat = [
            G.degree(node),
            nx.clustering(G, node),
            # Add your features here
        ]
        features.append(feat)
    
    # Convert to PyTorch Geometric format
    # ... (see data_loader.py for examples)
    
    return G, features, edge_index
```

## Output

Results are saved in `results/`:
- **Models**: `results/models/best_agn_model.pth`
- **Generated Nodes**: `results/generated/generated_nodes.csv`
- **Plots**: `results/plots/` (network comparisons, PCA, novelty analysis)

## Dependencies

See `requirements.txt` for complete list. Main dependencies:
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0
- NetworkX >= 2.6.0
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0

## Notes

- The code is designed to work with **any network type**
- No domain-specific dependencies (no RDKit, no chemical libraries)
- Easy to extend with custom datasets and evaluation metrics
- Clean, modular structure suitable for research and publication
