# GNNBlockDTI — Colab & GPU Compatible

This repository contains a reconstructed, faithful implementation of the GNNBlockDTI model (Deng et al., 2025) for drug-target interaction prediction. It is designed to be fully runnable on Google Colab with GPU acceleration and is 100% compatible with the official pre-trained checkpoints.

## Key Features
- **Faithful Architecture**: 100% compatible with the official PyTorch checkpoints (`strict=True` loading).
- **Comprehensive Test Suite**: Over 100 tests verifying structural and functional parity with the original codebase.
- **Memory Optimized**: The preprocessing pipeline (ESM-1b and ProtBERT) includes memory management specifically tailored for large sequences, preventing OOM errors.
- **Colab Ready**: Includes a Jupyter Notebook and specific requirements for seamless execution on Google Colab.

## Getting Started on Google Colab

The easiest way to run the training or testing pipeline is via Google Colab.
1. Open [`notebooks/GNNBlockDTI_Colab.ipynb`](notebooks/GNNBlockDTI_Colab.ipynb) in Google Colab.
2. Enable GPU (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Run the cells sequentially. The notebook handles dynamic DGL installation based on the Colab environment.

## Local Setup

### 1. Requirements

Ensure you have a modern Python 3.10+ environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Checkpoints

To ensure your environment correctly loads the official checkpoints:
```bash
python scripts/check_checkpoints.py
```

### 3. Preprocessing Data

If you want to preprocess the BIOSNAP data from raw SMILES and sequences using ESM-1b and ProtBERT:
```bash
python scripts/preprocess_biosnap.py --device cuda
```
*Note: This takes about ~30 minutes on a T4 GPU and requires ~4GB of GPU RAM.*

### 4. Training

To train the model using cross-validation fold 0:
```bash
python scripts/train.py --fold 0 --epochs 100
```

### 5. Evaluation

To evaluate an existing checkpoint on fold 0:
```bash
python scripts/test_biosnap.py --checkpoint official/pre_train_models/BIOSNAP_CV1.pth --fold 0
```

## Documentation
- [`docs/CODE_AUDIT.md`](docs/CODE_AUDIT.md): Detailed audit of the reconstructed code vs. the official implementation.
- [`docs/IMPLEMENTATION_MAPPING.md`](docs/IMPLEMENTATION_MAPPING.md): Mapping of official files/variables to this repository.
- [`docs/COLAB_RUN.md`](docs/COLAB_RUN.md): Step-by-step Colab setup guide.
