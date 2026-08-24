# GNNBlockDTI Code Audit & Repair Log

## Official Codebase Analysis
The original GNNBlockDTI official repository (commit `531356f66703dd52ab371a6387453eb49da7760d`) contained several implementation bugs, inconsistencies with the paper, and runtime errors.

## Bugs Identified & Repaired

### 1. `models.py`
- **Missing Import**: `GATConv` is used in `GATGCN_Block` but was never imported from DGL.
- **Undefined Class `GCN_Block`**: Line 133 instantiates `GCN_Block`, which is never defined in the code.
  - *Repair*: Forensic analysis of the pre-trained checkpoints revealed `attn_l` and `attn_r` parameters in `net_D.gcn_1`, proving the initial block uses GAT layers. Thus, `GCN_Block` was intended to be `GATGCN_Block`.
- **Misnamed Internal Modules**: The code names internal modules `GNNBlock0`, `GNNBlocks`, but the checkpoint expects `gcn_1`, `gcn_2s`.
  - *Repair*: Renamed modules in `GNNBlocks` class to match checkpoint state_dict keys exactly.
- **Pooling Typo**: Code instantiates `self.maxpool = AvgPooling()` inside `GNNBlocks`, which contradicts its variable name and the paper diagram.
  - *Repair*: Kept `AvgPooling()` as implemented (checkpoint doesn't reveal pooling type as it's parameter-free, so source code implementation takes precedence).

### 2. `Trainer.py`
- **Uninitialized Variable**: `valid_L` list is appended to inside the loop but never initialized before the loop.
- **Variable Typo**: Print statement references `valid__L` instead of `valid_L`.
  - *Repair*: Fixed both initialization and typo in reconstructed `trainer.py`.

### 3. `test.py`
- **Shadowing**: Imports `test` function from `Trainer` but then defines a local function named `test()`, shadowing the import.
- **Undefined variables**:
  - Uses `n` instead of function parameter `k`.
  - References `mydataset` and `collate_fn` which are not imported.
  - Returns undefined `best_metric`.
  - References `prot_seq` and `prot_data` instead of `target_embedding` and `target_graph`.
  - *Repair*: Fully reconstructed `scripts/test.py` resolving all scoping and import issues.

### 4. `main.py`
- **Missing Import**: `dgl` is not imported but used in `collate_fn`.
- **Undefined Variables**:
  - References `drug_data` instead of `drug_graph` in `mydataset`.
  - References `prot_seq` and `prot_data` instead of embeddings and graphs.
  - *Repair*: Fixed in `src/data/dataset.py` and `scripts/train.py`.

### 5. `data_process.py`
- **Inconsistent Device Usage**: Short protein sequences were processed on GPU, but sequences > 1000 were processed on CPU.
  - *Repair*: Consolidated ESM-1b processing in `target_graph_construct` to consistently use the provided torch device.

## Checkpoint Fidelity
By repairing the structural bugs in `models.py` (specifically changing `GCN_Block` to `GATGCN_Block` and correcting module names), the reconstructed architecture perfectly aligns with the official pre-trained checkpoints (`BIOSNAP_CV1.pth`, `BIOSNAP_unseen_D.pth`, `BIOSNAP_unseen_T.pth`) with exactly 3,342,978 parameters.
