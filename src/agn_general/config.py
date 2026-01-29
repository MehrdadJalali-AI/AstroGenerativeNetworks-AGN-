"""
Configuration file for Generalized AGN (Astro Generative Network)
Works with any network type
"""

import torch
import os

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory paths - go up from src/agn_general/ to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
GENERATED_DIR = os.path.join(RESULTS_DIR, "generated")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, MODEL_DIR, PLOTS_DIR, GENERATED_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model hyperparameters
HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_GCN_LAYERS = 2

# Training parameters
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Generation parameters
NUM_GENERATED_NODES = 100  # Number of nodes to generate
K_NEIGHBORS = 10  # Number of nearest neighbors to connect
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity for edge creation

# Dataset parameters
DATASET_PORTION = 1.0  # Use full dataset (1000+ nodes)
RANDOM_SEED = 42
