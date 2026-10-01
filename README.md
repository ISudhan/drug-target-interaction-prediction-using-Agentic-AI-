<h1 align="center">🧬 GNNBlockDTI</h1>
<p align="center">
  <b>Drug–Target Interaction Prediction via Gated Graph Neural Networks &amp; Multi-modal Protein Encoding</b><br/>
  A faithful, production-quality reconstruction of <em>Deng et al., 2025</em> — fully compatible with the official pre-trained checkpoints.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1.2-red?logo=pytorch" alt="PyTorch 2.1.2"/>
  <img src="https://img.shields.io/badge/DGL-1.1.2-orange" alt="DGL 1.1.2"/>
  <img src="https://img.shields.io/badge/Dataset-BIOSNAP-green" alt="BIOSNAP"/>
  <img src="https://img.shields.io/badge/Tests-100%2B-brightgreen?logo=pytest" alt="100+ Tests"/>
  <img src="https://img.shields.io/badge/Checkpoint-strict%3DTrue-blueviolet" alt="Checkpoint compatible"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [Drug Branch — GNNBlocks](#1-drug-branch--gnnblocks)
  - [Protein Branch — MultiscaleCNN + WGCN](#2-protein-branch--multiscalecnn--wgcn)
  - [Fusion Strategies](#3-fusion-strategies)
  - [Classifier Head](#4-classifier-head)
  - [Dimension Flow](#dimension-flow)
- [Data Pipeline](#data-pipeline)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
  - [Step 1 — Preprocessing](#step-1--preprocessing)
  - [Step 2 — Training](#step-2--training)
  - [Step 3 — Evaluation](#step-3--evaluation)
  - [Step 4 — Streamlit App](#step-4--streamlit-app)
  - [Step 5 — Tests](#step-5--tests)
- [Configuration Reference](#configuration-reference)
- [Key Implementation Notes](#key-implementation-notes)
- [Metrics](#metrics)
- [Google Colab](#google-colab)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Overview

GNNBlockDTI is a bimodal deep learning framework for **binary drug–target interaction (DTI) prediction**. It jointly encodes:

- **Drugs** as molecular graphs via stacked, gated Graph Neural Network blocks (GAT + GCN with GRU-like gating units).
- **Proteins** via two parallel branches — a multi-scale 1D CNN over ProtBERT logit embeddings, and a Weighted GCN over ESM-1b contact-map graphs.

The two modalities are fused and classified by a 3-layer MLP producing interaction/no-interaction logits.

This implementation is a clean-room reconstruction with:
- **100% strict checkpoint compatibility** (`strict=True` weight loading against official `.pth` files)
- **Memory-safe preprocessing** — incremental ESM-1b, chunked processing for proteins > 500 residues
- **100+ unit/integration tests** verifying structural and functional parity
- **Streamlit inference app** for interactive, zero-setup predictions

---

## Model Architecture

```
Drug SMILES ──► smi_2_graph ──► [N, 64]
                                     │
                               GNNBlocks (GAT+GCN+GU)
                                     │
                               drug_emb [B, 192] ──────────────────────────┐
                                                                            │
Protein Seq ──► ProtBERT logits ──► [L, 30]                                │
                                     │                                      │
                               MultiscaleCNN (3-branch CNN)           Concat [B, 192+fused]
                                     │                                      │
                               p_seq [B, L, 192] ──► Fusion ──► p_fused    │
                                     │                    │                  │
                               WGCN (3-layer GCN)         │           MLP (1024→1024→512→2)
                                     │                    │                  │
                               p_graph [B, L, 256] ───────┘            logits [B, 2]
                                                                            │
Contact Map ──► ESM-1b ─────────────────────────────────────────     Softmax → P(interact)
```

---

### 1. Drug Branch — GNNBlocks

**File:** [`src/models/gnn_blocks.py`](src/models/gnn_blocks.py)

The drug molecule (SMILES → 64-dim atom features) is encoded through `L+1` stacked **GNNBlocks** followed by average pooling.

#### GATGCN_Block

Each block combines:
1. **`n-1` GAT layers** (Graph Attention Network, 4 heads) — captures multi-hop substructural patterns.
2. **1 GCN layer** — feature enhancement, doubles the hidden dimension.

Returns both the GAT output `h` and GCN-enhanced output `h_1` for downstream residual computation.

```
input → GAT¹ → ... → GAT^(n-1)  → h_gat  [N, hidden]
                                → GCN^n  → h_gcn  [N, hidden*2]
```

#### Gated_NN (Gating Unit)

A GRU-inspired gating mechanism placed between consecutive blocks to filter redundant information:

```
R  = σ(h₀·Wᵣ₁ + h₁·Wᵣ₂ + bᵣ)             ← Reset gate
Z  = σ(h₀·W_z₁ + h₁·W_z₂ + b_z)           ← Update gate
h₂ = tanh(h₁·W_h₁ + (R⊙h₀)·W_h₂ + b_h)  ← Candidate hidden
h  = Z⊙h₀ + (1-Z)⊙h₂                     ← Gated output
```

A single `Gated_NN` instance is **shared** across all block transitions (not per-block copies).

#### GNNBlocks (Full Stack)

```
GNNBlock₀(input=64, hidden=96, out=192, layers=3)   → h₁ [N, 192]
    │
for i in 0..L-1:
    GATGCN_Block(hidden=192, out=384, layers=2)
    Residual: h₂ = (h_gat + dropout(GLU(h_gcn))) × 1/√2
    LayerNorm(h₂)
    GatingUnit(h_prev, h₂) → h_new [N, 192]
    │
AvgPooling → drug_emb [batch, 192]
```

Default: `L=5` subsequent blocks, `n_gnn=2` layers per block.

> **Paper vs. Code discrepancy:** The paper diagram labels the pooling as "Maxpool", but the official code uses `dgl.nn.AvgPooling()`. This implementation matches the official code.

---

### 2. Protein Branch — MultiscaleCNN + WGCN

**File:** [`src/models/protein.py`](src/models/protein.py)

Two parallel sub-networks encode the same protein via different modalities:

#### MultiscaleCNN (Sequence Branch)

Takes per-residue **ProtBERT logits** `[batch, L, 30]` through a cascaded 3-branch CNN:

```
Linear(30 → 128) → permute → [B, 128, L]
    │
    ├─ Conv1D(128→32,  k=5,  pad='same') → ReLU → FC(32→32)   ┐
    │  out: [B, L, 32]                                         │
    ├─ Conv1D(32→64,   k=7,  pad='same') → ReLU → FC(64→64)   ├─ Concat → [B, L, 192]
    │  out: [B, L, 64]                                         │
    └─ Conv1D(64→96,   k=13, pad='same') → ReLU → FC(96→96)   ┘
       out: [B, L, 96]
```

Output: `p_seq [B, L, 192]`  (32 + 64 + 96 = 192)

> Note: The three branches are **sequential**, not strictly parallel — branch N feeds into branch N+1. This matches the official code despite the paper's parallel diagram.

#### WGCN (Contact Map Branch)

Takes the **ESM-1b contact map** as a DGL graph. Node features are ProtBERT logits `[L, 30]`, edge weights are contact probabilities:

```
EdgeWeightNorm(both)
→ GCN(30  → 128, edge_weight) → LayerNorm(128)
→ GCN(128 → 128, edge_weight) → LayerNorm(128)
→ GCN(128 → 256, edge_weight) → LayerNorm(256)
Output: node embeddings [N_total, 256]
```

After WGCN, node embeddings are split per-protein and zero-padded to the max sequence length in the batch, producing `p_graph [B, L, 256]`.

---

### 3. Fusion Strategies

**File:** [`src/models/protein.py`](src/models/protein.py)

Three strategies merge `p_seq [B, L, 192]` and `p_graph [B, L, 256]`, selected via `--fusion` flag:

| Strategy | Mechanism | Output Dim |
|---|---|---|
| `concat` *(default)* | FC each to 384 → L2-norm → concatenate → AdaptiveMaxPool1d(1) | **768** |
| `bilinear` | FC each to 384 → L2-norm → `nn.Bilinear(384,384,384)` → AdaptiveMaxPool1d(1) | **384** |
| `gated` | FC each to 384 → L2-norm → `σ(concat)` gate → blend → AdaptiveMaxPool1d(1) | **384** |

All strategies collapse the sequence dimension to a fixed-length protein vector via `AdaptiveMaxPool1d(1)`.

---

### 4. Classifier Head

**File:** [`src/models/model.py`](src/models/model.py)

```
[drug_emb(192) || protein_fused(768)]    # concat fusion → 960 dims total
    → Linear(960 → 1024) → ReLU → Dropout(0.2)
    → Linear(1024 → 1024) → ReLU → Dropout(0.2)
    → Linear(1024 → 512)
    → Linear(512 → 2)
    → logits [B, 2]   (class 0 = no interaction, class 1 = interaction)
```

> With `bilinear`/`gated` fusion: input to MLP is 192 + 384 = **576 dims**.

---

### Dimension Flow

| Stage | Shape |
|---|---|
| Atom feature extraction | `[N_atoms, 64]` |
| After GNNBlocks + AvgPool | `[B, 192]` |
| ProtBERT logits | `[B, L, 30]` |
| After MultiscaleCNN | `[B, L, 192]` |
| After WGCN | `[N_total, 256]` → padded `[B, L, 256]` |
| After ConcatenationFusion | `[B, 768]` |
| After MLP | `[B, 2]` |

---

## Data Pipeline

```
Raw SMILES  ──► smi_2_graph()              ──► drug_graph.pkl
                RDKit mol parsing
                64-dim atom feats
                (symbol|charge|degree|aromatic|ring)
                dgl.to_bidirected()

Protein Seq ──► get_protbert_embedding()   ──► target_embedding.pkl
                Rostlab/prot_bert_bfd
                output = model(**ids).logits[0][1:-1]  ← logits, NOT hidden states
                shape: [L, 30]

            ──► target_graph_construct()   ──► target_distance.pkl
                ESM-1b (esm1b_t33_650M_UR50S)
                return_contacts=True
                Long proteins (L>500): chunked overlap=250, averaged
                Resumable: saves every 10 proteins

            ──► get_target_graph()         ──► target_graph.pkl
                Contact edges: prob > 0.5 (weight = raw probability)
                Backbone edges: i↔i+1     (weight = 1.0)
                Backbone always overrides contact for adjacent residues

CV Splits   ──► data/dataset/BIOSNAP/random_CV5/data_CV{0..4}.pkl
                Each file: (train_list, valid_list, test_list)
                Each item: [drug_id, target_id, label]
```

---

## Project Structure

```
.
├── app.py                          # Streamlit interactive inference app
├── configs/
│   └── biosnap.py                  # BIOSNAPConfig dataclass (all hyperparameters)
├── data/
│   └── dataset/BIOSNAP/            # Preprocessed features (auto-generated)
│       ├── drug_graph.pkl
│       ├── target_embedding.pkl
│       ├── target_distance.pkl     # ESM-1b contact maps (intermediate)
│       ├── target_graph.pkl
│       └── random_CV5/
│           └── data_CV{0..4}.pkl   # 5-fold CV splits
├── docs/
│   ├── CODE_AUDIT.md               # Every discrepancy vs. official code, with justifications
│   ├── IMPLEMENTATION_MAPPING.md   # Official file → this repo variable mapping
│   └── COLAB_RUN.md                # Google Colab step-by-step guide
├── models/                         # Saved .pt checkpoints (created at training time)
├── notebooks/
│   └── GNNBlockDTI_Colab.ipynb     # End-to-end Colab notebook
├── scripts/
│   ├── preprocess_biosnap.py       # Preprocessing CLI (Step 1)
│   ├── train.py                    # Training CLI (Step 2)
│   ├── test_biosnap.py             # Evaluation CLI (Step 3)
│   └── check_checkpoints.py        # Verify checkpoint key/shape compatibility
├── src/
│   ├── data/
│   │   ├── dataset.py              # DTIDataset + collate_fn
│   │   └── preprocessing.py        # SMILES→graph, ProtBERT, ESM-1b pipelines
│   ├── evaluation/
│   │   └── evaluate.py             # estimate() → AUROC, AUPR, ACC, P, R, F1
│   ├── models/
│   │   ├── gnn_blocks.py           # GATGCN_Block, Gated_NN, GNNBlocks
│   │   ├── protein.py              # WGCN, MultiscaleCNN, [Concat/Bilinear/Gated]Fusion
│   │   └── model.py                # GNNBlockDTI (full model, forward pass)
│   └── training/
│       └── trainer.py              # train() / test() loops, checkpointing
├── tests/                          # 100+ pytest tests
│   ├── test_models.py              # Architecture, checkpoint loading
│   ├── test_forward.py             # End-to-end forward pass
│   ├── test_dataset.py             # Dataset + collate_fn
│   ├── test_preprocessing.py       # Feature extraction correctness
│   ├── test_checkpoint.py          # strict=True weight key verification
│   └── test_official_data_layout.py
├── requirements.txt
├── requirements-colab.txt
└── pytest.ini
```

---

## Environment Setup

> **Python 3.10+ required.** GPU strongly recommended for preprocessing and training.

### Local (venv)

```bash
# 1. Clone the repository
git clone <repo-url>
cd Drug-target-interaction-prediction-using-Agentic-AI

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.1.2 | Core deep learning |
| `dgl` | 1.1.2 | Graph neural network layers |
| `transformers` | 4.35.2 | ProtBERT (`Rostlab/prot_bert_bfd`) |
| `fair-esm` | 2.0.0 | ESM-1b contact map prediction |
| `rdkit` | ≥2022.9.1 | SMILES → molecular graph |
| `scikit-learn` | ≥1.1.0 | Evaluation metrics |

> **DGL GPU wheel:** If DGL import fails on GPU, manually install the CUDA-matched build:
> ```bash
> pip install dgl -f https://data.dgl.ai/wheels/repo.html
> ```

---

## Running the Project

### Step 1 — Preprocessing

Converts raw SMILES strings and protein sequences into model-ready features.

**Required input files** (place in `official/dataset/BIOSNAP/`):
- `drug_smi_raw.pkl` — `dict[drug_id → smiles_string]`
- `prot_seq_raw.pkl` — `dict[protein_id → amino_acid_sequence]`

```bash
python scripts/preprocess_biosnap.py --device cuda
```

| Flag | Default | Description |
|---|---|---|
| `--task` | `BIOSNAP` | Dataset name |
| `--data_dir` | `official/dataset/BIOSNAP` | Path to raw data directory |
| `--device` | `cuda` if available | `cuda` or `cpu` |
| `--force` | `False` | Recompute even if output `.pkl` files already exist |

**Outputs written to `--data_dir`:**
```
drug_graph.pkl          ← DGL molecular graphs per drug
target_embedding.pkl    ← ProtBERT logits [L, 30] per protein
target_distance.pkl     ← ESM-1b contact maps (incremental checkpoint)
target_graph.pkl        ← DGL contact-map graphs per protein
```

> ⏱ **Runtime:** ~30 minutes on NVIDIA T4 GPU.  
> 💾 **GPU RAM:** ~4 GB peak.  
> 🔄 **Resumable:** The ESM-1b step checkpoints progress every 10 proteins. Re-running automatically resumes from the last saved state.

---

### Step 2 — Training

```bash
python scripts/train.py \
    --fold 0 \
    --epochs 100 \
    --fusion concat \
    --device cuda:0
```

| Flag | Default | Description |
|---|---|---|
| `--fold` | `0` | Cross-validation fold index (0–4) |
| `--epochs` | `100` (from config) | Number of training epochs |
| `--batch_size` | `64` (from config) | Batch size |
| `--lr` | `5e-4` (from config) | Learning rate (Adam) |
| `--fusion` | `concat` | Protein fusion strategy: `concat`, `bilinear`, `gated` |
| `--device` | auto-detect | Target device (`cuda:0`, `cpu`) |
| `--data_dir` | `official/dataset/BIOSNAP` | Preprocessed feature directory |
| `--save_dir` | `models/` | Checkpoint output directory |

**Model selection:** The checkpoint with the best **validation AUROC** is saved. Test metrics are logged whenever a new best is found.

**Output:**
```
models/GNNBlockDTI_BIOSNAP_CV0.pt
```

**Example training output:**
```
Loading data features for BIOSNAP...
Loading CV fold 0 splits...
Data train: 33772  Data valid: 4222  Data test: 4222
Running on cuda:0
epoch: 1 train_loss: 0.621 valid_loss: 0.594
The best model in epoch 1 has been saved!!!
test metric (AUROC, AUPR, ACC, Precision, Recall, F1): (0.85231, 0.82174, 0.79012, ...)
...
Best Test Metrics - AUROC: 0.91234, AUPR: 0.88123, ACC: 0.84567, ...
```

---

### Step 3 — Evaluation

Evaluate a saved checkpoint on a specific CV fold:

```bash
python scripts/test_biosnap.py \
    --checkpoint models/GNNBlockDTI_BIOSNAP_CV0.pt \
    --fold 0
```

Verify checkpoint compatibility with official pre-trained weights:

```bash
python scripts/check_checkpoints.py
```

This script confirms that all weight keys and tensor shapes exactly match the official checkpoint — essential before reporting results against the paper's numbers.

---

### Step 4 — Streamlit App

Interactive web interface for real-time DTI inference. No training required — loads saved `.pt` weights automatically.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**Mode 1 — Known Dataset IDs (fast lookup):**
```
Drug ID:    DB08533     (BIOSNAP DrugBank ID)
Target ID:  P49862      (UniProt accession)
```
Looks up pre-processed drug/target features from the cached `.pkl` files.

**Mode 2 — Raw SMILES + Protein Sequence (any molecule):**
```
SMILES:   CC1=C(C=C(C=C1)NC(=O)...)
Sequence: MVLSPADKTNVKAAWGKVGAHAGE...
```
- Drug graph is built on-the-fly from SMILES via RDKit.
- ProtBERT is loaded once (cached) to generate `[L, 30]` logit embeddings.
- Contact graph uses backbone-only edges (i ↔ i+1) as a lightweight approximation.

**Output:**
- Interaction probability, no-interaction probability
- Binary verdict (INTERACT / NO INTERACTION) with color-coded confidence

> **Accuracy note:** For maximum accuracy on unseen proteins, run the full ESM-1b preprocessing pipeline first to generate true contact maps. Backbone-only graphs are an approximation.

---

### Step 5 — Tests

```bash
# Run the full test suite
pytest

# Verbose mode
pytest -v

# Specific test module
pytest tests/test_forward.py -v
pytest tests/test_models.py -v
pytest tests/test_preprocessing.py -v

# Skip slow tests (e.g., ProtBERT loading)
pytest -m "not slow"
```

**Test coverage:**

| File | What is tested |
|---|---|
| `test_models.py` | Architecture shapes, checkpoint `strict=True` loading, parameter counts |
| `test_forward.py` | Full forward pass with synthetic data, output shape `[B, 2]` |
| `test_dataset.py` | `DTIDataset` filtering, `collate_fn` batching, padding logic |
| `test_preprocessing.py` | SMILES parsing, 64-dim atom features, contact map edge construction |
| `test_checkpoint.py` | Weight key and shape verification against official `.pth` file |
| `test_official_data_layout.py` | Dataset file structure and pickle schema validation |

---

## Configuration Reference

All hyperparameters in [`configs/biosnap.py`](configs/biosnap.py):

```python
@dataclass
class BIOSNAPConfig:
    # ── Training ────────────────────────────────────
    epochs:      int   = 100       # Training epochs
    batch_size:  int   = 64        # Batch size
    lr:          float = 5e-4      # Learning rate (Adam)
    seed:        int   = 1234      # Reproducibility seed

    # ── Model Architecture ──────────────────────────
    n_layer:     int   = 5         # Number of subsequent GNNBlocks (L)
    n_gnn:       int   = 2         # GNN layers per block (1 GAT + 1 GCN)
    hidden_size: int   = 128       # WGCN hidden dim
    embed_dim:   int   = 128       # MultiscaleCNN initial embedding dim
    hid_size:    int   = 384       # Fusion projection dim

    # ── Dropout ─────────────────────────────────────
    dropout_d:   float = 0.2       # Drug branch (GNNBlocks)
    dropout_DT:  float = 0.2       # Fusion + classifier

    # ── Derived Dimensions (match checkpoint) ───────
    drug_len:    int   = 192       # GNNBlocks output dim  (96 × 2)
    target_len:  int   = 192       # MultiscaleCNN output  (32 + 64 + 96)
    target_len1: int   = 256       # WGCN output dim       (128 × 2)
```

CLI flags passed to `train.py` override config values at runtime.

---

## Key Implementation Notes

Non-obvious decisions that differ from the paper or the variable names used in the official code:

### 1. ProtBERT Logits, Not Hidden States
```python
# CORRECT — used in this implementation
output = model(input_ids=ids, attention_mask=mask)
embedding = output[0][0][1:-1]   # logits, shape [L, 30]

# WRONG — do NOT use this
embedding = output.hidden_states[-1][0][1:-1]  # shape [L, 1024]
```
Using hidden states breaks checkpoint compatibility and changes the feature distribution entirely.

### 2. `self.maxpool` Is Actually `AvgPooling`

```python
# In GNNBlocks — despite the variable name, this is AvgPooling
self.maxpool = AvgPooling()   # matches official code behavior
```

### 3. Gating Unit Is Shared Across All Block Transitions

One `Gated_NN` instance handles all `L` inter-block transitions. There is no per-block gating weights.

### 4. Drug Graphs Are Bidirected

```python
bg = dgl.to_bidirected(g)   # All molecular bonds become bidirectional
```
Required by GAT layers which expect symmetric message passing.

### 5. ESM-1b Chunking for Long Proteins

Proteins longer than 500 residues are split into overlapping chunks (size=500, step=250). Contact predictions from overlapping regions are averaged:

```python
contact_prob_map[start:end, start:end] += chunk_contacts
count_map[start:end, start:end] += 1.0
final_map = contact_prob_map / count_map   # average overlaps
```

### 6. `gcn_1` Contains GAT Parameters

Despite being named `gcn_1`, the initial block in the checkpoint contains `attn_l` / `attn_r` parameters — proving it is a `GATGCN_Block`, not a plain GCN. Loading any other block type here will fail `strict=True` checkpoint loading.

---

## Metrics

All metrics computed in [`src/evaluation/evaluate.py`](src/evaluation/evaluate.py) via softmax probabilities of class 1:

| Metric | Role |
|---|---|
| **AUROC** | Primary model selection metric (maximize) |
| **AUPR** | Area Under Precision-Recall Curve — more informative for imbalanced data |
| **Accuracy** | Binary classification accuracy |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | Harmonic mean of Precision and Recall |

Reported values are **test-set metrics at the epoch with the best validation AUROC**.

---

## Google Colab

The fastest path to the full pipeline without a local GPU:

1. Open [`notebooks/GNNBlockDTI_Colab.ipynb`](notebooks/GNNBlockDTI_Colab.ipynb) in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells sequentially — the notebook auto-detects the Colab environment and installs the correct DGL wheel

To install Colab-specific requirements manually:
```bash
pip install -r requirements-colab.txt
```

See [`docs/COLAB_RUN.md`](docs/COLAB_RUN.md) for the full step-by-step guide.

---

## Documentation

| Document | Description |
|---|---|
| [`docs/CODE_AUDIT.md`](docs/CODE_AUDIT.md) | Detailed audit of every discrepancy between this repo and the official codebase, with justifications for each decision |
| [`docs/IMPLEMENTATION_MAPPING.md`](docs/IMPLEMENTATION_MAPPING.md) | Maps each official file, class, and variable name to its equivalent in this repository |
| [`docs/COLAB_RUN.md`](docs/COLAB_RUN.md) | Step-by-step Google Colab setup and execution guide |

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{deng2025gnnblockdti,
  title   = {GNNBlock-DTI: Efficient substructure feature encoding based on
             graph neural network blocks for drug-target interaction prediction},
  author  = {Deng et al.},
  year    = {2025}
}
```

---

<p align="center">
  Built with PyTorch &middot; DGL &middot; ProtBERT &middot; ESM-1b &middot; RDKit &middot; Streamlit
</p>
