"""Test data preprocessing pipeline.

Verifies SMILES→graph conversion, contact map edge extraction,
and atom feature construction against the official implementation.

Tests requiring ProtBERT or ESM model downloads are marked @pytest.mark.slow.
"""
import os
import pickle
import pytest
import torch
import dgl
import numpy as np


# ════════════════════════════════════════════════════════════
#  SMILES → Graph conversion
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSmi2Graph:
    """Test drug molecular graph construction from SMILES."""

    @pytest.fixture
    def smi_2_graph(self):
        """Import smi_2_graph directly to avoid importing transformers."""
        from src.data.preprocessing import smi_2_graph
        return smi_2_graph

    @pytest.fixture
    def one_hot(self):
        from src.data.preprocessing import one_hot
        return one_hot

    def test_simple_smiles(self, smi_2_graph):
        """Test with a simple molecule (ethanol)."""
        g = smi_2_graph("CCO")
        assert g.num_nodes() > 0
        assert g.num_edges() > 0
        assert 'feat' in g.ndata

    def test_real_biosnap_smiles(self, smi_2_graph, biosnap_dir):
        """Test with the first SMILES from the official BIOSNAP dataset."""
        smi_path = os.path.join(biosnap_dir, "drug_smi_raw.pkl")
        if not os.path.exists(smi_path):
            pytest.skip("BIOSNAP drug_smi_raw.pkl not found")
        with open(smi_path, 'rb') as f:
            drugs = pickle.load(f)
        first_smi = list(drugs.values())[0]
        g = smi_2_graph(first_smi)
        assert g.num_nodes() > 0
        assert g.num_edges() > 0

    def test_atom_feature_dimension_is_64(self, smi_2_graph):
        """Official atom features: symbol(44) + charge(9) + degree(9) + aromatic(1) + ring(1) = 64."""
        g = smi_2_graph("CCO")
        assert g.ndata['feat'].shape[1] == 64, (
            f"Atom feature dim = {g.ndata['feat'].shape[1]}, expected 64"
        )

    def test_graph_is_bidirected(self, smi_2_graph):
        """Official code calls dgl.to_bidirected()."""
        g = smi_2_graph("CC")  # ethane: 1 bond → 2 edges after bidirection
        src, dst = g.edges()
        # For every edge (u, v), the reverse edge (v, u) should also exist
        edges = set(zip(src.tolist(), dst.tolist()))
        for u, v in list(edges):
            assert (v, u) in edges, f"Edge ({v}, {u}) missing — graph is not bidirected"

    def test_feature_dtype(self, smi_2_graph):
        g = smi_2_graph("c1ccccc1")  # benzene
        assert g.ndata['feat'].dtype == torch.float32

    def test_invalid_smiles_raises(self, smi_2_graph):
        with pytest.raises((ValueError, AttributeError)):
            smi_2_graph("INVALID_SMILES_STRING")

    def test_one_hot_known_element(self, one_hot):
        result = one_hot('C', ['C', 'N', 'O'])
        assert result[0] == 1.0
        assert result.sum() == 1.0
        assert len(result) == 4  # 3 + 1 (unknown slot)

    def test_one_hot_unknown_element(self, one_hot):
        result = one_hot('Xe', ['C', 'N', 'O'])
        assert result[-1] == 1.0  # unknown flag
        assert result.sum() == 1.0

    def test_batch_drug_graph_processing(self, biosnap_dir):
        """Test processing multiple SMILES like get_drug_graph does."""
        smi_path = os.path.join(biosnap_dir, "drug_smi_raw.pkl")
        if not os.path.exists(smi_path):
            pytest.skip("BIOSNAP drug_smi_raw.pkl not found")

        from src.data.preprocessing import smi_2_graph
        with open(smi_path, 'rb') as f:
            drugs = pickle.load(f)

        # Test first 10 drugs
        success, fail = 0, 0
        for drug_id in list(drugs.keys())[:10]:
            try:
                g = smi_2_graph(drugs[drug_id])
                assert g.num_nodes() > 0
                assert g.ndata['feat'].shape[1] == 64
                success += 1
            except Exception:
                fail += 1

        assert success > 0, "No drug SMILES could be processed"
        assert success >= 9, f"Too many failures: {fail}/10"


# ════════════════════════════════════════════════════════════
#  Contact map edge extraction (get_uv)
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetUV:

    @pytest.fixture
    def get_uv(self):
        from src.data.preprocessing import get_uv
        return get_uv

    def test_sequential_edges_always_present(self, get_uv):
        """Adjacent residues (i-1, i+1) always have edges with weight 1.0."""
        adj = np.zeros((5, 5))  # No contacts above threshold
        u, v, w = get_uv(adj)
        # Sequential edges: (0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)
        edges = set(zip(u, v))
        for i in range(4):
            assert (i, i + 1) in edges, f"Missing sequential edge ({i}, {i+1})"
            assert (i + 1, i) in edges, f"Missing sequential edge ({i+1}, {i})"

    def test_sequential_edge_weight_is_1(self, get_uv):
        adj = np.zeros((3, 3))
        u, v, w = get_uv(adj)
        for ui, vi, wi in zip(u, v, w):
            if abs(ui - vi) == 1:
                assert wi == 1.0, f"Sequential edge ({ui},{vi}) weight = {wi}, expected 1.0"

    def test_contact_threshold_0_5(self, get_uv):
        """Only contacts > 0.5 get edges."""
        adj = np.array([
            [0.0, 0.3, 0.8],
            [0.3, 0.0, 0.5],  # 0.5 is NOT > 0.5
            [0.8, 0.5, 0.0],
        ])
        u, v, w = get_uv(adj)
        edges = list(zip(u, v))
        # (0,2) and (2,0) should exist as contact edges
        assert (0, 2) in edges, "Contact edge (0,2) with prob 0.8 should exist"
        assert (2, 0) in edges, "Contact edge (2,0) with prob 0.8 should exist"
        # (0,1) and (1,0) exist as sequential edges only (prob 0.3 < 0.5)
        # (1,2) and (2,1) exist as sequential edges only (prob 0.5 NOT > 0.5)

    def test_contact_weight_equals_probability(self, get_uv):
        """Non-sequential contact edges use the actual probability as weight."""
        adj = np.zeros((5, 5))
        adj[0][3] = 0.75
        adj[3][0] = 0.75
        u, v, w = get_uv(adj)
        for ui, vi, wi in zip(u, v, w):
            if (ui, vi) == (0, 3) or (ui, vi) == (3, 0):
                assert abs(wi - 0.75) < 1e-6, f"Contact weight should be 0.75, got {wi}"

    def test_self_loops_exist_when_diagonal_above_threshold(self, get_uv):
        """Official get_uv creates self-loops when adj[i][i] > 0.5.
        This matches the official code behavior (no diagonal exclusion).
        """
        adj = np.ones((5, 5))  # All contacts including diagonal
        u, v, w = get_uv(adj)
        self_loops = [(ui, vi) for ui, vi in zip(u, v) if ui == vi]
        # With adj[i][i] = 1.0 > 0.5 and i != i-1 and i != i+1, self-loops ARE created
        assert len(self_loops) > 0, "Expected self-loops when diagonal > 0.5"

    def test_no_self_loops_when_diagonal_zero(self, get_uv):
        """No self-loops when diagonal is zero."""
        adj = np.zeros((5, 5))
        adj[0, 2] = 0.8
        adj[2, 0] = 0.8
        u, v, w = get_uv(adj)
        self_loops = [(ui, vi) for ui, vi in zip(u, v) if ui == vi]
        assert len(self_loops) == 0, f"Unexpected self-loops: {self_loops}"


# ════════════════════════════════════════════════════════════
#  ProtBERT Embedding (slow — requires model download)
# ════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.integration
class TestProtBERTEmbedding:
    """These tests download the ProtBERT model (~1.7GB) and are skipped by default.

    Run with: pytest -m slow
    """

    def test_protbert_output_dimension(self):
        """ProtBERT logits should be 30-dimensional (vocab size for amino acids)."""
        from src.data.preprocessing import get_protbert_embedding
        data = {"test_prot": "ACDEFGHIKLMNPQRSTVWY"}  # 20 standard amino acids
        result = get_protbert_embedding(data, device=torch.device('cpu'))
        assert "test_prot" in result
        assert result["test_prot"].shape == (20, 30), (
            f"Expected (20, 30), got {result['test_prot'].shape}"
        )

    def test_protbert_excludes_cls_sep(self):
        """Output length should equal input sequence length (CLS/SEP removed)."""
        from src.data.preprocessing import get_protbert_embedding
        seq = "ACDEF"
        data = {"test": seq}
        result = get_protbert_embedding(data, device=torch.device('cpu'))
        assert result["test"].shape[0] == len(seq)


# ════════════════════════════════════════════════════════════
#  Padding
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPadding:

    @pytest.fixture
    def padding(self):
        from src.data.preprocessing import padding
        return padding

    def test_short_sequence_padded(self, padding):
        data = {"p1": torch.randn(10, 30)}
        result = padding(data, maxlen=50)
        assert result["p1"].shape == (50, 30)
        # Original data preserved
        assert torch.allclose(result["p1"][:10], data["p1"])
        # Padding is zeros
        assert (result["p1"][10:] == 0).all()

    def test_long_sequence_truncated(self, padding):
        data = {"p1": torch.randn(100, 30)}
        result = padding(data, maxlen=50)
        assert result["p1"].shape == (50, 30)

    def test_exact_length_preserved(self, padding):
        data = {"p1": torch.randn(50, 30)}
        result = padding(data, maxlen=50)
        assert result["p1"].shape == (50, 30)
