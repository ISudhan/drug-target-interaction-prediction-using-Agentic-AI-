"""Test official checkpoint compatibility with the reconstructed model.

These are the authoritative checkpoint tests — they use REAL official checkpoints
with strict=True loading. No mock DGL classes.

Requires: official/pre_train_models/*.pth
"""
import os
import pytest
import torch

from src.models.gnn_blocks import GNNBlocks
from src.models.protein import WGCN, MultiscaleCNN
from src.models.model import GNNBlockDTI
from configs.biosnap import BIOSNAPConfig


# ════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════

CHECKPOINT_NAMES = [
    "BIOSNAP_CV1.pth",
    "BIOSNAP_unseen_D.pth",
    "BIOSNAP_unseen_T.pth",
]


@pytest.fixture(params=CHECKPOINT_NAMES)
def checkpoint_path_and_name(request, checkpoint_dir):
    """Parametrized fixture: yields (path, name) for each checkpoint."""
    name = request.param
    path = os.path.join(checkpoint_dir, name)
    if not os.path.exists(path):
        pytest.skip(f"Checkpoint not found: {path}")
    return path, name


@pytest.fixture
def loaded_state_dict(checkpoint_path_and_name):
    path, _ = checkpoint_path_and_name
    return torch.load(path, map_location='cpu', weights_only=False)


# ════════════════════════════════════════════════════════════
#  Strict loading
# ════════════════════════════════════════════════════════════

@pytest.mark.checkpoint
class TestStrictCheckpointLoading:
    """These tests verify that strict=True loading works for all official checkpoints."""

    def test_strict_load_succeeds(self, build_model, checkpoint_path_and_name):
        """Core test: strict=True loading must pass with no errors."""
        path, name = checkpoint_path_and_name
        model = build_model()
        state_dict = torch.load(path, map_location='cpu', weights_only=False)

        # This is the definitive test — if strict=True fails, the architecture is wrong
        result = model.load_state_dict(state_dict, strict=True)

        assert len(result.missing_keys) == 0, (
            f"{name}: Missing keys: {result.missing_keys}"
        )
        assert len(result.unexpected_keys) == 0, (
            f"{name}: Unexpected keys: {result.unexpected_keys}"
        )


# ════════════════════════════════════════════════════════════
#  Key verification
# ════════════════════════════════════════════════════════════

@pytest.mark.checkpoint
class TestCheckpointKeyStructure:

    def test_no_missing_keys(self, build_model, loaded_state_dict):
        model = build_model()
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(loaded_state_dict.keys())
        missing = model_keys - ckpt_keys
        assert len(missing) == 0, f"Missing keys: {missing}"

    def test_no_unexpected_keys(self, build_model, loaded_state_dict):
        model = build_model()
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(loaded_state_dict.keys())
        unexpected = ckpt_keys - model_keys
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    def test_key_count_is_97(self, loaded_state_dict):
        """All official checkpoints have exactly 97 keys."""
        assert len(loaded_state_dict) == 97, (
            f"Expected 97 keys, got {len(loaded_state_dict)}"
        )


# ════════════════════════════════════════════════════════════
#  Shape verification
# ════════════════════════════════════════════════════════════

@pytest.mark.checkpoint
class TestCheckpointShapes:

    def test_no_shape_mismatch(self, build_model, loaded_state_dict):
        """Every key's tensor shape must match between model and checkpoint."""
        model = build_model()
        model_sd = model.state_dict()
        for key in loaded_state_dict:
            if key not in model_sd:
                continue
            assert loaded_state_dict[key].shape == model_sd[key].shape, (
                f"Shape mismatch for {key}: "
                f"checkpoint={loaded_state_dict[key].shape}, "
                f"model={model_sd[key].shape}"
            )

    def test_classifier_weight_shape(self, loaded_state_dict):
        """fc.weight should be [2, 512]."""
        assert loaded_state_dict['fc.weight'].shape == (2, 512)
        assert loaded_state_dict['fc.bias'].shape == (2,)

    def test_initial_gat_shapes(self, loaded_state_dict):
        """Initial block GAT layer 0: fc.weight [96, 64] (input=64, output per head=24, 4 heads)."""
        assert loaded_state_dict['net_D.gcn_1.net.0.fc.weight'].shape == (96, 64)
        assert loaded_state_dict['net_D.gcn_1.net.0.attn_l'].shape == (1, 4, 24)
        assert loaded_state_dict['net_D.gcn_1.net.0.attn_r'].shape == (1, 4, 24)

    def test_pair_net_input_dimension(self, loaded_state_dict):
        """pair_net.0.weight: [1024, 576] — 576 = Drug_len(192) + hid_size(384)."""
        assert loaded_state_dict['pair_net.0.weight'].shape == (1024, 576)

    def test_fusion_fc_shapes(self, loaded_state_dict):
        """FF.fc1: [384, 192], FF.fc2: [384, 256]."""
        assert loaded_state_dict['FF.fc1.weight'].shape == (384, 192)
        assert loaded_state_dict['FF.fc2.weight'].shape == (384, 256)


# ════════════════════════════════════════════════════════════
#  Parameter count
# ════════════════════════════════════════════════════════════

@pytest.mark.checkpoint
class TestCheckpointParameterCount:

    def test_total_parameters(self, loaded_state_dict):
        """All checkpoints should have 3,342,978 total parameters."""
        total = sum(v.numel() for v in loaded_state_dict.values())
        assert total == 3_342_978, f"Parameter count {total} != 3,342,978"

    def test_model_matches_checkpoint_params(self, build_model, loaded_state_dict):
        model = build_model()
        model_params = sum(p.numel() for p in model.parameters())
        ckpt_params = sum(v.numel() for v in loaded_state_dict.values())
        assert model_params == ckpt_params, (
            f"Model params ({model_params}) != checkpoint params ({ckpt_params})"
        )
