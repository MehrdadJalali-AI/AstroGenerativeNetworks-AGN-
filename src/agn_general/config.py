"""
Configuration for Astro Generative Network (AGN)
"""

import os
import torch

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory paths - project root (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
GENERATED_DIR = os.path.join(RESULTS_DIR, "generated")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
# Final IEEE figures live at repository root for easy collection
PAPER_FIGURES_DIR = BASE_DIR

for dir_path in [DATA_DIR, RESULTS_DIR, MODEL_DIR, PLOTS_DIR, GENERATED_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model hyperparameters
HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_GCN_LAYERS = 2

# Training parameters (override with env AGN_EPOCHS for quick local runs)
EPOCHS = int(os.environ.get("AGN_EPOCHS", "200"))
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Loss weights (Option B full objective)
BETA = 1.0   # KL weight
GAMMA = 1.0  # feature reconstruction weight

# Early stopping: monitor validation loss when enabled
EARLY_STOPPING_PATIENCE = 20
USE_VALIDATION_EARLY_STOPPING = True

# Link split fractions (train / val / test)
LINK_SPLIT_NUM_VAL = 0.1
LINK_SPLIT_NUM_TEST = 0.1

# Generation parameters
NUM_GENERATED_NODES = 100
K_NEIGHBORS = 10
SIMILARITY_THRESHOLD = 0.5

# Small real graphs: cap generated nodes
NUM_GENERATED_NODES_SMALL = 15

# Dataset parameters
DATASET_PORTION = 1.0
RANDOM_SEED = 42

# Default experiment grid (synthetic + optional real)
EXPERIMENT_DATASETS = [
    "karate",
    "facebook",
    "email",
    "sbm_assortative",
    "sbm_disassortative",
    "sbm_core_periphery",
    "scale_free_sparse",
]
# Datasets used for main paper triptychs (match manuscript regime names)
PAPER_TRIPTYCH_DATASETS = ["karate", "facebook", "scale_free_sparse"]
# Extra real-world topology experiments (appendix-ready)
REAL_EXPERIMENT_DATASETS = ["zachary", "lesmis"]

INSERTION_VARIANTS = ["original", "no_gg", "strict_gg", "density_adaptive", "density_adaptive_no_gg"]
