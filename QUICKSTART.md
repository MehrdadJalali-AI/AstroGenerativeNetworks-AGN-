# Quick Start Guide - Generalized AGN

## Installation

1. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

2. **Verify installation:**
```bash
python3 -c "import torch; import torch_geometric; import networkx; print('All dependencies installed!')"
```

## Running AGN

### Option 1: Run from project root
```bash
python3 run_agn.py
```

### Option 2: Run from module directory
```bash
cd src/agn_general
python3 main.py
```

## What It Does

The script will:
1. Load three social network datasets (Karate Club, Facebook Ego, Email Network)
2. Use 30% of each dataset (configurable in `config.py`)
3. Train a VGAE model on each network
4. Generate 10 new nodes for each network (configurable)
5. Insert generated nodes with appropriate edges
6. Evaluate and visualize results

## Output

Results are saved in `results/`:
- **`models/best_agn_model.pth`**: Trained model
- **`generated/generated_nodes.csv`**: Generated node features
- **`plots/`**: Visualization plots

## Configuration

Edit `src/agn_general/config.py` to customize:

```python
# Dataset portion (0.0 to 1.0)
DATASET_PORTION = 0.3  # Use 30% of dataset

# Number of nodes to generate
NUM_GENERATED_NODES = 10

# Number of neighbors to connect
K_NEIGHBORS = 5

# Similarity threshold for edges
SIMILARITY_THRESHOLD = 0.5

# Training epochs
EPOCHS = 200
```

## Expected Runtime

- Small networks (Karate Club): ~1-2 minutes
- Medium networks (Facebook, Email): ~3-5 minutes each
- Total: ~10-15 minutes for all three datasets

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root or have added `src` to your Python path.

### CUDA Errors
The code automatically uses CPU if CUDA is not available. To force CPU:
```python
# In config.py
DEVICE = torch.device("cpu")
```

### Memory Issues
If you run out of memory:
1. Reduce `DATASET_PORTION` (e.g., 0.2 instead of 0.3)
2. Reduce `NUM_GENERATED_NODES`
3. Reduce `EPOCHS`

## Next Steps

1. **Try different datasets**: Modify `data_loader.py` to add your own networks
2. **Adjust hyperparameters**: Experiment with `HIDDEN_DIM`, `LATENT_DIM`, etc.
3. **Customize evaluation**: Modify `evaluation.py` to add your own metrics
