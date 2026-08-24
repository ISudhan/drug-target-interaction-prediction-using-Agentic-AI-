"""Root conftest for GNNBlockDTI test suite.

Provides shared fixtures and path setup.
"""
import os
import sys
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# ── Directories ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(__file__)
OFFICIAL_DIR = os.path.join(PROJECT_ROOT, "official")
CHECKPOINT_DIR = os.path.join(OFFICIAL_DIR, "pre_train_models")
BIOSNAP_DIR = os.path.join(OFFICIAL_DIR, "dataset", "BIOSNAP")


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def official_dir():
    return OFFICIAL_DIR


@pytest.fixture
def checkpoint_dir():
    return CHECKPOINT_DIR


@pytest.fixture
def biosnap_dir():
    return BIOSNAP_DIR


@pytest.fixture
def checkpoint_paths():
    """Return paths to all 3 official pretrained checkpoints."""
    return [
        os.path.join(CHECKPOINT_DIR, "BIOSNAP_CV1.pth"),
        os.path.join(CHECKPOINT_DIR, "BIOSNAP_unseen_D.pth"),
        os.path.join(CHECKPOINT_DIR, "BIOSNAP_unseen_T.pth"),
    ]


@pytest.fixture
def biosnap_config():
    """Return a BIOSNAPConfig instance."""
    from configs.biosnap import BIOSNAPConfig
    return BIOSNAPConfig()


@pytest.fixture
def build_model(biosnap_config):
    """Factory fixture: builds a GNNBlockDTI model with default BIOSNAP config."""
    def _build():
        from src.models.gnn_blocks import GNNBlocks
        from src.models.protein import WGCN, MultiscaleCNN
        from src.models.model import GNNBlockDTI

        config = biosnap_config
        net_D = GNNBlocks(
            input_size=64, hidden_size=96,
            n_layer=config.n_layer, n_gnn=config.n_gnn,
            dropout=config.dropout_d
        )
        net_T = MultiscaleCNN(
            inputs=30, embed_dim=config.embed_dim,
            nums_conv=[32, 64, 96], ksize=[5, 7, 13],
            dropout=0.2
        )
        net_T1 = WGCN(input_size=30, hidden_size=config.hidden_size, dropout=0.2)
        model = GNNBlockDTI(
            net_D, net_T, net_T1,
            Drug_len=config.drug_len,
            Target_len=config.target_len,
            Target_len1=config.target_len1,
            hid_size=config.hid_size,
            dropout=config.dropout_DT
        )
        return model
    return _build
