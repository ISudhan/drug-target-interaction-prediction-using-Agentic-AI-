"""Test DTIDataset and collate function.

Verifies dataset construction, sample integrity, silent sample loss accounting,
and correct batching behavior.
"""
import pytest
import torch
import dgl

from src.data.dataset import DTIDataset, collate_fn


# ════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════

def _make_drug_graph(num_nodes=5):
    """Helper: create a synthetic drug graph with 64-dim features."""
    src = torch.randint(0, num_nodes, (num_nodes * 2,))
    dst = torch.randint(0, num_nodes, (num_nodes * 2,))
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata['feat'] = torch.randn(num_nodes, 64)
    return g


def _make_target_graph(num_nodes=50):
    """Helper: create a synthetic protein contact graph with 30-dim features."""
    u, v, w = [], [], []
    for i in range(num_nodes - 1):
        u.extend([i, i + 1])
        v.extend([i + 1, i])
        w.extend([1.0, 1.0])
    g = dgl.graph((u, v), num_nodes=num_nodes)
    g.ndata['feat'] = torch.randn(num_nodes, 30)
    g.edata['weight'] = torch.tensor(w, dtype=torch.float32)
    return g


@pytest.fixture
def synthetic_data():
    """Create synthetic drug graphs, target embeddings/graphs, and interaction data."""
    seq_len = 50
    drug_graph = {
        "D001": _make_drug_graph(5),
        "D002": _make_drug_graph(8),
        "D003": _make_drug_graph(3),
    }
    target_embedding = {
        "T001": torch.randn(seq_len, 30),
        "T002": torch.randn(seq_len, 30),
    }
    target_graph = {
        "T001": _make_target_graph(seq_len),
        "T002": _make_target_graph(seq_len),
    }
    # Interaction data: (drug_id, target_id, label)
    data = [
        ("D001", "T001", 1.0),
        ("D002", "T001", 0.0),
        ("D003", "T002", 1.0),
        ("D001", "T002", 0.0),
    ]
    return data, drug_graph, target_embedding, target_graph


# ════════════════════════════════════════════════════════════
#  Dataset construction
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDTIDatasetConstruction:

    def test_dataset_length(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"

    def test_item_structure(self, synthetic_data):
        """Each item should be (drug_graph, target_embedding, target_graph, label)."""
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        D, T, Tg, Y = dataset[0]
        assert isinstance(D, dgl.DGLGraph), "Drug should be a DGL graph"
        assert isinstance(T, torch.Tensor), "Target embedding should be a tensor"
        assert isinstance(Tg, dgl.DGLGraph), "Target graph should be a DGL graph"
        assert isinstance(Y, int), "Label should be an integer"

    def test_labels_are_valid(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        for i in range(len(dataset)):
            _, _, _, y = dataset[i]
            assert y in (0, 1), f"Label at index {i} is {y}, expected 0 or 1"

    def test_drug_features_correct_dim(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        D, _, _, _ = dataset[0]
        assert D.ndata['feat'].shape[1] == 64


# ════════════════════════════════════════════════════════════
#  Silent sample loss accounting
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDatasetSampleLossAccounting:
    """The dataset silently skips samples when features are missing.

    This is inherited from the official code's `except: pass` pattern.
    These tests expose the behavior and verify it's only happening for
    genuinely missing keys, not for unexpected errors.
    """

    def test_missing_drug_skipped(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        # Add an entry with a drug ID that doesn't exist
        data_with_missing = data + [("D_MISSING", "T001", 1.0)]
        dataset = DTIDataset(data_with_missing, dg, te, tg)
        assert len(dataset) == len(data), (
            f"Expected {len(data)} samples (1 skipped), got {len(dataset)}"
        )

    def test_missing_target_embedding_skipped(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        data_with_missing = data + [("D001", "T_MISSING", 1.0)]
        dataset = DTIDataset(data_with_missing, dg, te, tg)
        assert len(dataset) == len(data)

    def test_missing_target_graph_skipped(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        # Remove T002 from target_graph only
        tg_partial = {"T001": tg["T001"]}
        # 2 of 4 entries reference T002 → they will be skipped
        dataset = DTIDataset(data, dg, te, tg_partial)
        expected = sum(1 for _, tid, _ in data if tid in tg_partial)
        assert len(dataset) == expected

    def test_sample_loss_accounting(self, synthetic_data):
        """Explicit accounting: total requested vs successfully loaded."""
        data, dg, te, tg = synthetic_data
        n_missing_drug = 3
        n_missing_target = 2

        extended_data = list(data)
        for i in range(n_missing_drug):
            extended_data.append((f"D_FAKE_{i}", "T001", 0.0))
        for i in range(n_missing_target):
            extended_data.append(("D001", f"T_FAKE_{i}", 1.0))

        total_requested = len(extended_data)
        dataset = DTIDataset(extended_data, dg, te, tg)
        successfully_loaded = len(dataset)
        samples_lost = total_requested - successfully_loaded

        assert total_requested == len(data) + n_missing_drug + n_missing_target
        assert samples_lost == n_missing_drug + n_missing_target, (
            f"Expected {n_missing_drug + n_missing_target} lost samples, got {samples_lost}"
        )
        assert successfully_loaded == len(data)

    def test_no_unexpected_loss_with_complete_data(self, synthetic_data):
        """When all features are present, NO samples should be lost."""
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        assert len(dataset) == len(data), (
            f"Unexpected sample loss: {len(data) - len(dataset)} samples disappeared"
        )


# ════════════════════════════════════════════════════════════
#  Collate function
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCollateFn:

    def test_collate_produces_correct_types(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        batch = [dataset[i] for i in range(len(dataset))]
        D_G, T, T_G, Y = collate_fn(batch)

        assert isinstance(D_G, dgl.DGLGraph), "Batched drug should be DGL graph"
        assert isinstance(T, torch.Tensor), "Batched target embedding should be tensor"
        assert isinstance(T_G, dgl.DGLGraph), "Batched target graph should be DGL graph"
        assert isinstance(Y, torch.Tensor), "Labels should be tensor"

    def test_collate_shapes(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        batch = [dataset[i] for i in range(len(dataset))]
        D_G, T, T_G, Y = collate_fn(batch)

        batch_size = len(data)
        assert T.shape[0] == batch_size, f"T batch dim should be {batch_size}"
        assert T.shape[2] == 30, "T feature dim should be 30"
        assert Y.shape == (batch_size,), f"Y shape should be ({batch_size},)"

    def test_collate_label_dtype(self, synthetic_data):
        data, dg, te, tg = synthetic_data
        dataset = DTIDataset(data, dg, te, tg)
        batch = [dataset[i] for i in range(len(dataset))]
        _, _, _, Y = collate_fn(batch)
        assert Y.dtype == torch.long, f"Labels dtype should be torch.long, got {Y.dtype}"
