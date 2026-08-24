#!/bin/bash
set -e
uv venv --python 3.10 .venv_audit
source .venv_audit/bin/activate
uv pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install dgl -f https://data.dgl.ai/wheels/repo.html
uv pip install transformers numpy scikit-learn rdkit pandas
# We don't need ESM just to load the state dict in inspect_checkpoint.py
python scripts/inspect_checkpoint.py
