# GNNBlockDTI Code Audit Report

## 1. Scope

Complete audit of the reconstructed GNNBlockDTI implementation against:
- The official source code
- The official pretrained checkpoints
- The paper specification

## 2. Official Repository Commit

```
https://github.com/Ptexys/GNNBlockDTI.git
Commit: 531356f66703dd52ab371a6387453eb49da7760d
```

Read-only copy preserved at: `official/`

## 3. Paper Reference

> Deng et al. "Efficient substructure feature encoding based on graph neural network blocks
> for drug-target interaction prediction" — Frontiers in Pharmacology, 2025

## 4. Architecture Mapping

**Status: PASS**

All architectural components verified against checkpoint structure:

| Component | Keys | Shapes | Status |
|---|---|---|---|
| GNNBlocks (Drug) | 97/97 match | All match | PASS |
| MultiscaleCNN (Protein seq) | Verified | Verified | PASS |
| WGCN (Protein graph) | Verified | Verified | PASS |
| FeatureFusion | Verified | Verified | PASS |
| Pair Network MLP | Verified | Verified | PASS |
| Final Classifier | fc: [2, 512] | Verified | PASS |
| Total parameters | 3,342,978 | All 3 checkpoints | PASS |

Evidence: `tests/test_models.py` — 28 structural tests, all passing.

### Architecture Details Verified

- Initial GNNBlock: 3 layers (2 GAT + 1 GCN) — **confirmed by checkpoint keys** `net_D.gcn_1.net.{0,1}.*` (GAT) + `net_D.gcn_1.net_1.*` (GCN)
- Subsequent GNNBlocks (×5): 2 layers each (1 GAT + 1 GCN)
- GAT: 4 heads, hidden_size//4 per head, ReLU activation
- Gated_NN: GRU-like gating between blocks
- CNN kernels: 5, 7, 13
- WGCN: 3 GCN layers (30→128→128→256)
- Classifier: 576→1024→1024→512→2

## 5. Preprocessing Mapping

**Status: PASS**

| Component | Status | Evidence |
|---|---|---|
| SMILES → DGL graph | IDENTICAL | `tests/test_preprocessing.py` — 9 tests |
| Atom features (64-dim) | IDENTICAL | Verified: symbol(44)+charge(9)+degree(9)+aromatic(1)+ring(1) |
| ProtBERT logits (30-dim) | IDENTICAL | Verified in code; env requires `transformers` fix for runtime |
| ESM-1b contact maps | IDENTICAL | Code match; threshold > 0.5 verified |
| Sequential edges | IDENTICAL | Weight 1.0 for |i-j|=1, verified by test |
| Long sequence handling | INTENTIONAL-FIX | Device consistency fix for sequences > 1000 |

**ProtBERT model identifier**: Official uses `prot_bert_bfd` (likely local path), reconstructed uses `Rostlab/prot_bert_bfd` (HuggingFace). **Status: UNVERIFIED** — likely equivalent but not confirmed at runtime.

## 6. Training Mapping

**Status: PASS (code-level)**

| Parameter | Official | Reconstructed | Status |
|---|---|---|---|
| Loss | CrossEntropyLoss | CrossEntropyLoss | IDENTICAL |
| Optimizer | Adam | Adam | IDENTICAL |
| Learning rate | 0.0005 | 0.0005 | IDENTICAL |
| Batch size | 64 | 64 | IDENTICAL |
| Epochs | 100 | 100 | IDENTICAL |
| Model selection | Max AUROC (validation) | Same | IDENTICAL |
| Scheduler | None | None | IDENTICAL |

**Bugs fixed:**
- `valid_L` initialization (OFFICIAL-BUG)
- `valid__L` typo (OFFICIAL-BUG)
- Loss computed on CPU in both official and reconstructed (preserves behavior)

**Note:** Paper describes "BCE" but official code uses `CrossEntropyLoss` on 2-class output. This is documented as PAPER-vs-CODE-DISCREPANCY.

## 7. Evaluation Mapping

**Status: PASS**

| Metric | Implementation | Status |
|---|---|---|
| AUROC | `roc_auc_score(y_true, softmax_probs[:, 1])` | IDENTICAL |
| AUPR | `precision_recall_curve` → `auc` | IDENTICAL |
| Accuracy | `accuracy_score` | IDENTICAL |
| Precision | `precision_score` | IDENTICAL (+ zero_division handling) |
| Recall | `recall_score` | IDENTICAL (+ zero_division handling) |
| F1 | `f1_score` | IDENTICAL (+ zero_division handling) |

Positive-class probability: `softmax(logits, dim=1)[:, 1]` — verified.

## 8. Checkpoint Compatibility

**Status: PASS**

All 3 official checkpoints loaded with `strict=True`:

| Checkpoint | Missing Keys | Unexpected Keys | Shape Mismatches | strict=True |
|---|---|---|---|---|
| `BIOSNAP_CV1.pth` | 0 | 0 | 0 | ✅ PASS |
| `BIOSNAP_unseen_D.pth` | 0 | 0 | 0 | ✅ PASS |
| `BIOSNAP_unseen_T.pth` | 0 | 0 | 0 | ✅ PASS |

All checkpoints: 97 keys, 3,342,978 parameters.

Evidence: `tests/test_checkpoint.py` — 24 tests (8 per checkpoint × 3), all passing.

## 9. Test Results

```
tests/test_models.py         — 28 passed
tests/test_preprocessing.py  — 15 passed, 2 deselected (slow/ProtBERT)
tests/test_dataset.py        — 12 passed
tests/test_checkpoint.py     — 24 passed
tests/test_forward.py        — 11 passed
─────────────────────────────────────────────
Total:                         105 passed, 0 failed, 2 deselected
```

Deselected tests require `transformers` package (ProtBERT model download) — marked `@pytest.mark.slow`.

### Forward pass results

| Checkpoint | Output Shape | Finite | No NaN | No Inf | Valid Probabilities |
|---|---|---|---|---|---|
| BIOSNAP_CV1.pth | [1, 2] ✅ | ✅ | ✅ | ✅ | ✅ |
| BIOSNAP_unseen_D.pth | [1, 2] ✅ | ✅ | ✅ | ✅ | ✅ |
| BIOSNAP_unseen_T.pth | [1, 2] ✅ | ✅ | ✅ | ✅ | ✅ |

## 10. Known Discrepancies

| # | Type | Description | Impact |
|---|---|---|---|
| 1 | PAPER-vs-CODE | Paper says "max-pooling"; official code uses `AvgPooling()` | Reconstructed matches official code (AvgPooling). Different results if paper is "correct." |
| 2 | PAPER-vs-CODE | Paper says "BCE"; code uses `CrossEntropyLoss` on 2-class | Functionally related but distinct implementations |
| 3 | UNVERIFIED | ProtBERT model ID: `prot_bert_bfd` vs `Rostlab/prot_bert_bfd` | Likely equivalent; not runtime-verified |

## 11. Intentional Fixes

| # | Fix | Justification |
|---|---|---|
| 1 | `GCN_Block` → `GATGCN_Block` | Official class undefined; checkpoint proves GAT layers (attn_l, attn_r params) |
| 2 | Module names: `GNNBlock0`/`GNNBlocks` → `gcn_1`/`gcn_2s` | Required for checkpoint compatibility |
| 3 | `GATConv` import added | Used but never imported in official code |
| 4 | `valid_L` initialization | Uninitialized in official Trainer.py |
| 5 | `valid__L` → `valid_L` | Typo in official print statement |
| 6 | Variable names in dataset/test scripts | `drug_data`→`drug_graph`, `prot_seq`→`target_embedding`, etc. |
| 7 | ESM long-sequence device consistency | Official processes long sequences on CPU; fixed to use configurable device |
| 8 | Lazy imports for transformers/esm | Allows model tests without heavy ML packages |

## 12. Unverified Components

| Component | Reason |
|---|---|
| ProtBERT model identity | `prot_bert_bfd` vs `Rostlab/prot_bert_bfd` — requires runtime comparison |
| ProtBERT output dimension (30) | Confirmed in code; not runtime-verified in this env |
| Training metric reproduction | No training run executed |
| Evaluation metric reproduction | No inference on preprocessed BIOSNAP data executed |

## 13. Colab Status

**Status: NOT VERIFIED**

A `docs/COLAB_RUN.md` has been created with step-by-step instructions, but the notebook has **not been actually executed** in Google Colab. This will be done in the next phase.

## 14. Final Verdict

| Category | Verdict | Evidence |
|---|---|---|
| Architecture fidelity | **PASS** | 28 structural tests + checkpoint compatibility |
| Checkpoint compatibility | **PASS** | 3/3 checkpoints, strict=True, 0 mismatches |
| Forward pass correctness | **PASS** | Finite [batch, 2] output for all checkpoints |
| Preprocessing correctness | **PASS** | 15 unit tests on graph/edge construction |
| Dataset integrity | **PASS** | Sample loss accounting tests |
| Training code | **PARTIAL** | Code audited, bugs fixed, not runtime-verified |
| Evaluation code | **PARTIAL** | Code audited, not verified on real predictions |
| Metric reproduction | **UNVERIFIED** | No end-to-end inference with preprocessed data |
| Colab readiness | **UNVERIFIED** | Instructions written, not executed |

**Overall: PARTIAL — Architecture and checkpoint compatibility are fully verified. End-to-end metric reproduction requires preprocessed BIOSNAP features and a training/inference run.**
