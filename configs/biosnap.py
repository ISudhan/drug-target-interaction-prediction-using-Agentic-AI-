"""
Configuration for BIOSNAP dataset experiments.
Matches the official paper hyperparameters and settings.
"""

import os
from dataclasses import dataclass


@dataclass
class BIOSNAPConfig:
    # Training hyperparams
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.0005
    
    # Model architecture hyperparams
    n_layer: int = 5          # Number of GNNBlocks after initial block
    n_gnn: int = 2            # GNN layers per subsequent block (1 GAT + 1 GCN)
    
    # Dropout
    dropout_d: float = 0.2    # Drug branch dropout
    dropout_DT: float = 0.2   # Fusion/Classifier dropout
    
    # Fixed Dimensions (from paper / checkpoint)
    hidden_size: int = 128    # Used for WGCN 
    embed_dim: int = 128      # Used for MultiscaleCNN embedding
    hid_size: int = 384       # Used for Feature Fusion
    
    # Feature Dimensions
    drug_len: int = 192       # Drug embedding dimension (96 * 2)
    target_len: int = 192     # Target sequence CNN dimension (32 + 64 + 96)
    target_len1: int = 256    # Target graph WGCN dimension (hidden_size * 2)
    
    # Paths
    dataset_name: str = "BIOSNAP"
    data_dir: str = os.path.join("official/dataset", "BIOSNAP")
    preprocessed_dir: str = data_dir  # By default, preprocessed features are in the same dir
    random_cv_dir: str = os.path.join(data_dir, "random_CV5")
    
    seed: int = 1234
