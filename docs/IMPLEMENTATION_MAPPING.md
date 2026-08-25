# GNNBlockDTI — Implementation Mapping

Official Repository: [Ptexys/GNNBlockDTI](https://github.com/Ptexys/GNNBlockDTI) @ commit `531356f`
Paper: "Efficient substructure feature encoding based on graph neural network blocks for drug-target interaction prediction" (Deng et al., Frontiers in Pharmacology, 2025)

---

## File-Level Mapping

| Official File | Reconstructed File(s) | Role |
|---|---|---|
| `models.py` | `src/models/gnn_blocks.py`, `src/models/protein.py`, `src/models/model.py` | Model architecture |
| `data_process.py` | `src/data/preprocessing.py`, `scripts/preprocess_biosnap.py` | Data preprocessing |
| `main.py` | `scripts/train.py`, `src/data/dataset.py` | Training pipeline + dataset class |
| `Trainer.py` | `src/training/trainer.py`, `src/evaluation/evaluate.py` | Training loop + metrics |
| `test.py` | `scripts/test_biosnap.py` | Evaluation script |

---

## Component-Level Mapping

### Drug Branch: GNNBlocks

| Component | Official (`models.py`) | Reconstructed (`gnn_blocks.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| Class name | `GNNBlocks` | `GNNBlocks` | GNNBlock | IDENTICAL | — |
| Initial block class | `GCN_Block` (undefined!) | `GATGCN_Block` | "GNNBlock₀" | **OFFICIAL-BUG** | Checkpoint contains `attn_l`, `attn_r` params in `net_D.gcn_1.net.0`, proving GAT layers. `GCN_Block` is undefined in official code. |
| Initial block layers | `nums_layer=3` (2 GAT + 1 GCN) | `nums_layer=3` (2 GAT + 1 GCN) | Figure 1b subscript₃ | IDENTICAL | Checkpoint keys: `gcn_1.net.0.*`, `gcn_1.net.1.*`, `gcn_1.net_1.*` |
| Subsequent blocks | `GATGCN_Block`, `nums_layer=n_gnn` (2) | `GATGCN_Block`, `nums_layer=n_gnn` (2) | GNNBlock₁..ₗ | IDENTICAL | Checkpoint keys: `gcn_2s.{i}.net.0.*`, `gcn_2s.{i}.net_1.*` |
| n_layer (L) | 5 | 5 | "L GNNBlocks" | IDENTICAL | From `Arguments()` class |
| n_gnn | 2 | 2 | — | IDENTICAL | From `Arguments()` class |
| Module name: initial | `self.GNNBlock0` | `self.gcn_1` | — | **REFACTORED-but-equivalent** | Official variable name won't match checkpoint; reconstruction uses checkpoint-compatible name |
| Module name: subsequent | `self.GNNBlocks` (variable) | `self.gcn_2s` | — | **REFACTORED-but-equivalent** | Same: reconstruction matches checkpoint key prefix |
| GAT n_heads | 4 | 4 | — | IDENTICAL | `n_heads=4` default |
| GAT activation | ReLU | ReLU | ReLU | IDENTICAL | — |
| GCN norm | 'both' | 'both' | — | IDENTICAL | — |
| Gated_NN | GRU-like gating | GRU-like gating | Eq. 1-4 | IDENTICAL | — |
| Residual connection | `(h1 + do(glu(h1_1))) * scale` | Same | Fig 1a | IDENTICAL | — |
| Scale factor | `sqrt(0.5)` | `sqrt(0.5)` | — | IDENTICAL | — |
| **Pooling** | `AvgPooling()` (named `self.maxpool`) | `AvgPooling()` | **"max-pooling layer"** | **PAPER-vs-CODE-DISCREPANCY** | Official code L151: `self.maxpool = AvgPooling()`. Paper text and Figure 1a both say "Maxpool". Checkpoint is parameter-free so either loads. **Reconstructed matches official code.** |
| GATConv import | **Missing** from imports | Present | — | **OFFICIAL-BUG** | Official `models.py` L8 imports from DGL but omits GATConv. Used in GATGCN_Block. |

### Protein Sequence Branch: MultiscaleCNN

| Component | Official | Reconstructed (`protein.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| Input dim | 30 (ProtBERT logits) | 30 | ProtBERT | IDENTICAL | `inputs=30` in main.py |
| Embed dim | 128 | 128 | — | IDENTICAL | `embed_dim=128` |
| Conv channels | [32, 64, 96] | [32, 64, 96] | — | IDENTICAL | — |
| Kernel sizes | [5, 7, 13] | [5, 7, 13] | "5, 7, and 13" | IDENTICAL | — |
| CNN ordering | Sequential (X→X1→X2→X3) | Sequential | — | IDENTICAL | Not parallel despite diagram |
| FC after each branch | Yes (fc1, fc2, fc3) | Yes | — | IDENTICAL | — |
| Output dim | 32+64+96=192 | 192 | — | IDENTICAL | — |

### Protein Graph Branch: WGCN

| Component | Official | Reconstructed (`protein.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| Input dim | 30 (ProtBERT logits) | 30 | — | IDENTICAL | — |
| GCN layers | 3 (30→128→128→256) | 3 (30→128→128→256) | — | IDENTICAL | Checkpoint shapes confirm |
| Edge weight norm | 'both' | 'both' | — | IDENTICAL | — |
| LayerNorm | After each GCN | After each GCN | — | IDENTICAL | — |
| GLU | Defined but unused in forward | Same | — | IDENTICAL | Dead code in both |
| maxpool | Defined but unused in forward | Same | — | IDENTICAL | Dead code in both |
| Output | Node embeddings (no pooling) | Same | — | IDENTICAL | Returns h3 directly |

### Feature Fusion

| Component | Official | Reconstructed (`protein.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| FC1 | Linear(192→384) | Same | — | IDENTICAL | Checkpoint |
| FC2 | Linear(256→384) | Same | — | IDENTICAL | Checkpoint |
| L2 normalize | Yes, p=2 dim=-1 | Yes | — | IDENTICAL | — |
| Fusion op | Element-wise add | Same | — | IDENTICAL | — |
| Pooling | AdaptiveMaxPool1d(1) | Same | — | IDENTICAL | — |

### Classifier (Pair Network)

| Component | Official | Reconstructed (`model.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| Input dim | 576 (192+384) | 576 | — | IDENTICAL | Checkpoint: pair_net.0.weight [1024, 576] |
| Architecture | 576→1024→ReLU→Dropout→1024→ReLU→Dropout→512 | Same | — | IDENTICAL | — |
| Final fc | Linear(512, 2) | Same | — | IDENTICAL | Checkpoint: fc.weight [2, 512] |
| Dropout rate | 0.2 | 0.2 | 0.2 | IDENTICAL | — |

### Loss & Training

| Component | Official (`Trainer.py`) | Reconstructed (`trainer.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| Loss function | `nn.CrossEntropyLoss()` | Same | **"BCE"** | **PAPER-vs-CODE-DISCREPANCY** | Paper says "binary cross-entropy" but code uses CrossEntropyLoss on 2-class logits. These are mathematically related but not identical implementations. |
| Optimizer | Adam | Adam | Adam | IDENTICAL | — |
| Learning rate | 0.0005 | 0.0005 | 5×10⁻⁴ | IDENTICAL | — |
| Batch size | 64 | 64 | 64 | IDENTICAL | — |
| Epochs | 100 | 100 | 100 | IDENTICAL | — |
| Scheduler | None | None | — | IDENTICAL | — |
| Model selection | Max AUROC on validation | Same | — | IDENTICAL | — |
| `valid_L` init | **Missing** | Fixed | — | **OFFICIAL-BUG** | Variable used but never initialized |
| `valid__L` typo | **Typo in print** | Fixed | — | **OFFICIAL-BUG** | Double underscore in f-string |

### Data Processing

| Component | Official (`data_process.py`) | Reconstructed (`preprocessing.py`) | Paper | Status | Evidence |
|---|---|---|---|---|---|
| ProtBERT model ID | `prot_bert_bfd` (local?) | `Rostlab/prot_bert_bfd` (HuggingFace) | ProtBERT-BFD | **UNVERIFIED** | Both likely resolve to the same model, but local path vs HuggingFace ID could differ |
| ProtBERT output | Logits (dim=30), exclude CLS/SEP | Same | — | IDENTICAL | `output[0][0][1:-1]` |
| ESM model | `esm1b_t33_650M_UR50S` | Same | ESM-1b | IDENTICAL | — |
| Contact threshold | > 0.5 | > 0.5 | — | IDENTICAL | — |
| Sequential edges | weight=1.0 for |i-j|=1 | Same | — | IDENTICAL | — |
| Long seq handling | Chunks of 500, overlap avg | Same | — | IDENTICAL | — |
| Long seq device | **CPU** (missing `.to(device)`) | Configurable device | — | **INTENTIONAL-FIX** | Official forgets `.to(device)` for batch_tokens in long-seq branch (L158) |
| Atom features | 64-dim (symbol+charge+degree+aromatic+ring) | Same | — | IDENTICAL | — |
| Drug graph | Bidirected | Same | — | IDENTICAL | — |

### Dataset

| Component | Official (`main.py`) | Reconstructed (`dataset.py`) | Status | Evidence |
|---|---|---|---|---|
| Variable names | `drug_data`, `prot_seq`, `prot_data` (all undefined) | `drug_graph`, `target_embedding`, `target_graph` | **OFFICIAL-BUG** | Official code references variables that don't exist in scope |
| Error handling | `except: pass` | `except KeyError: pass` | **REFACTORED-but-equivalent** | More precise exception type; same behavior |
| Label dtype | `int(ent[2])` implicit | Same, plus `dtype=torch.long` in collate | **REFACTORED-but-equivalent** | — |

### Evaluation

| Component | Official (`Trainer.py`) | Reconstructed (`evaluate.py`) | Status |
|---|---|---|---|
| Softmax | `F.softmax(Y_pred, 1)` | Same | IDENTICAL |
| AUROC | `roc_auc_score` | Same + ValueError handling | **REFACTORED-but-equivalent** |
| AUPR | `precision_recall_curve` + `auc` | Same | IDENTICAL |
| Metrics | AUROC, AUPR, ACC, Precision, Recall, F1 | Same | IDENTICAL |

---

## Summary of Discrepancies

| # | Type | Description |
|---|---|---|
| 1 | **OFFICIAL-BUG** | `GCN_Block` undefined — should be `GATGCN_Block` (proved by checkpoint) |
| 2 | **OFFICIAL-BUG** | `GATConv` used but never imported |
| 3 | **OFFICIAL-BUG** | `valid_L` uninitialized + `valid__L` typo in Trainer.py |
| 4 | **OFFICIAL-BUG** | `drug_data`, `prot_seq`, `prot_data` undefined in main.py/test.py |
| 5 | **OFFICIAL-BUG** | `test.py` shadows imported `test` function with local `test()` |
| 6 | **OFFICIAL-BUG** | `test.py` uses `n` instead of function param `k`; returns undefined `best_metric` |
| 7 | **PAPER-vs-CODE-DISCREPANCY** | Paper says "max-pooling", official code uses `AvgPooling()` |
| 8 | **PAPER-vs-CODE-DISCREPANCY** | Paper says "BCE", official code uses `CrossEntropyLoss` on 2-class output |
| 9 | **INTENTIONAL-FIX** | Long-sequence ESM processing moved to configurable device (official forgets `.to(device)`) |
| 10 | **UNVERIFIED** | ProtBERT model identifier: `prot_bert_bfd` vs `Rostlab/prot_bert_bfd` |
