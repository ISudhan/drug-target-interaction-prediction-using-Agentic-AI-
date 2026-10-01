# GNNBlockDTI — Google Colab Execution Guide

> ✅ **Colab Ready**: A Jupyter Notebook is provided in `notebooks/GNNBlockDTI_Colab.ipynb` to automate this workflow.

## Prerequisites

- Google Colab with GPU runtime (recommended: T4 or better)
- ~4GB GPU RAM for model inference and preprocessing
- ~8GB disk for ESM-1b + ProtBERT downloads (only needed for preprocessing)

---

## The Automated Way

Simply open `notebooks/GNNBlockDTI_Colab.ipynb` in Google Colab and run the cells. It handles cloning, dependency installation (including dynamic CUDA-matching for DGL), testing, preprocessing, training, and evaluation.

---

## The Manual Way

If you prefer to run commands manually in a Colab terminal or your own Jupyter notebook:

### Step 1: Clone Repository

```python
!git clone https://github.com/ISudhan/drug-target-interaction-prediction-using-Agentic-AI-.git
%cd drug-target-interaction-prediction-using-Agentic-AI-
```

### Step 2: Install Dependencies

Because Colab updates its PyTorch and CUDA versions frequently, we must install DGL dynamically.

```python
import torch
cuda_version = torch.version.cuda
cuda_version_str = "cu" + "".join(cuda_version.split(".")[:2])
dgl_install_cmd = f"pip install dgl -f https://data.dgl.ai/wheels/{cuda_version_str}/repo.html"
!{dgl_install_cmd}

!pip install -r requirements-colab.txt
```

### Step 3: Run Checkpoint Tests

Verify that the local model architecture correctly loads the official checkpoints.

```bash
!python scripts/check_checkpoints.py
```

Expected: All 3 checkpoints load with strict=True.

### Step 4: Run Fast Tests

```bash
!pytest tests/ -m "not slow" -q
```

Expected: All tests pass (except those skipped due to "slow" marker).

### Step 5: Full Preprocessing (~30 min on GPU)

```bash
!python scripts/preprocess_biosnap.py --device cuda
```
*Note: This script has memory optimizations built-in. It processes long sequences (>1000 aa) in chunks and aggressively garbage-collects tensors to avoid OOM on T4 GPUs.*

### Step 6: Training

```bash
!python scripts/train.py --fold 0 --epochs 100 --save_dir models_colab
```

### Step 7: Pretrained Inference (End-to-End)

Test the official checkpoint on fold 0 test split.

```bash
!python scripts/test_biosnap.py --checkpoint models/BIOSNAP_CV1.pth --fold 0
```

---

## Environment Notes

| Package | Tested Version | Notes |
|---|---|---|
| Python | 3.10 / 3.12 | Colab default |
| PyTorch | 2.1.2 / 2.3+ | Colab provides this |
| DGL | 1.1.2 / 2.3+ | Must match PyTorch/CUDA version |
| transformers | >=4.30.0 | For ProtBERT |
| fair-esm | >=2.0.0 | For ESM-1b contact maps |
| rdkit-pypi | >=2022.9 | For SMILES processing |
