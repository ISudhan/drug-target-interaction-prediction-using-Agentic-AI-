# GNNBlockDTI — Google Colab Execution Guide

> ⚠️ **NOT VERIFIED**: These instructions have NOT been executed in Colab yet.
> They are provided as a starting point. Actual verification will follow.

## Prerequisites

- Google Colab with GPU runtime (recommended: T4 or better)
- ~4GB GPU RAM for model inference
- ~8GB disk for ESM-1b + ProtBERT downloads (only needed for preprocessing)

---

## Step 1: Clone Repository

```python
!git clone https://github.com/ISudhan/drug-target-interaction-prediction-using-Agentic-AI-.git
%cd drug-target-interaction-prediction-using-Agentic-AI-
```

## Step 2: Install Dependencies

```bash
# Core dependencies (PyTorch usually pre-installed in Colab)
!pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
!pip install transformers==4.30.0
!pip install fair-esm
!pip install rdkit-pypi
!pip install pytest
```

## Step 3: Verify Package Versions

```python
import torch
import dgl
print(f"PyTorch: {torch.__version__}")
print(f"DGL: {dgl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import transformers
print(f"Transformers: {transformers.__version__}")

import rdkit
print(f"RDKit: {rdkit.__version__}")
```

## Step 4: Run Checkpoint Tests

```bash
# Quick smoke test — no pretrained model downloads needed
!python -m pytest tests/test_models.py tests/test_checkpoint.py -v
```

Expected: All tests pass, 3 checkpoints load with strict=True.

## Step 5: Run Forward Pass Tests

```bash
!python -m pytest tests/test_forward.py -v
```

Expected: Forward pass produces `[batch, 2]` output for all 3 checkpoints.

## Step 6: Run All Tests (excluding ProtBERT downloads)

```bash
!python -m pytest tests/ -v -m "not slow"
```

Expected: 105 passed, 2 deselected.

## Step 7: Pretrained Inference (End-to-End)

```python
import sys
sys.path.insert(0, '.')

import torch
import pickle
from src.models.gnn_blocks import GNNBlocks
from src.models.protein import WGCN, MultiscaleCNN
from src.models.model import GNNBlockDTI
from src.data.preprocessing import smi_2_graph
from configs.biosnap import BIOSNAPConfig

config = BIOSNAPConfig()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Build model
net_D = GNNBlocks(input_size=64, hidden_size=96, n_layer=5, n_gnn=2, dropout=0.2)
net_T = MultiscaleCNN(inputs=30, embed_dim=128, nums_conv=[32, 64, 96], ksize=[5, 7, 13], dropout=0.2)
net_T1 = WGCN(input_size=30, hidden_size=128, dropout=0.2)
model = GNNBlockDTI(net_D, net_T, net_T1, Drug_len=192, Target_len=192,
                    Target_len1=256, hid_size=384, dropout=0.2)

# Load checkpoint
checkpoint = torch.load('official/pre_train_models/BIOSNAP_CV1.pth',
                        map_location=device, weights_only=False)
model.load_state_dict(checkpoint, strict=True)
model.to(device)
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
print("Ready for inference.")
```

## Step 8: Full Preprocessing (Optional, ~30 min on GPU)

```python
# Only needed if you want to preprocess from raw SMILES/sequences
from src.data.preprocessing import (
    get_drug_graph, get_protbert_embedding,
    target_graph_construct, get_target_graph, padding
)

# Load raw data
with open('official/dataset/BIOSNAP/drug_smi_raw.pkl', 'rb') as f:
    drugs = pickle.load(f)
with open('official/dataset/BIOSNAP/prot_seq_raw.pkl', 'rb') as f:
    proteins = pickle.load(f)

# Process
drug_graphs = get_drug_graph(drugs)
target_embeddings = get_protbert_embedding(proteins)
target_distances = target_graph_construct(proteins)
target_graphs = get_target_graph(target_embeddings, target_distances)
```

## Step 9: Training (Optional)

```bash
# Requires preprocessed features from Step 8
!python scripts/train.py --save_name GNNBlockDTI_BIOSNAP_CV0
```

---

## Environment Notes

| Package | Tested Version | Notes |
|---|---|---|
| Python | 3.10 | Colab default |
| PyTorch | ≥2.1.0 | Colab provides this |
| DGL | 1.1.2 | Must match PyTorch version |
| transformers | 4.30.0 | Specific version to avoid tokenizers conflict |
| fair-esm | 2.0+ | For ESM-1b contact maps |
| rdkit-pypi | ≥2022.9 | For SMILES processing |

## Troubleshooting

### DGL Installation
If DGL fails to install, try:
```bash
!pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html  # Match CUDA version
```

### Tokenizers Conflict
If you see `tokenizers>=0.11.1,!=0.11.3,<0.14 is required`:
```bash
!pip install transformers==4.30.0 tokenizers==0.13.3
```
