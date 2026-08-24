"""Test real forward pass through the GNNBlockDTI model.

Uses a real BIOSNAP drug SMILES converted to a DGL graph and
synthetic protein data of correct dimensions.

For checkpoint tests, loads each official checkpoint and verifies
the forward pass produces finite [batch, 2] output.
"""
import os
import pickle
import pytest
import torch
import dgl

from src.models.gnn_blocks import GNNBlocks
from src.models.protein import WGCN, MultiscaleCNN
from src.models.model import GNNBlockDTI


# ════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════

def _smi_2_graph(smi):
    """Local copy of smi_2_graph to avoid importing transformers via src.data."""
    from rdkit import Chem

    def one_hot(char, dictionary):
        h = torch.zeros(len(dictionary) + 1)
        for i in range(len(dictionary)):
            if dictionary[i] == char:
                h[i] = 1
                break
            elif i == len(dictionary) - 1:
                h[-1] = 1
        return h

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    nums_atom = mol.GetNumAtoms()
    u, v = [], []
    for bond in mol.GetBonds():
        u.append(bond.GetBeginAtom().GetIdx())
        v.append(bond.GetEndAtom().GetIdx())
    u, v = torch.tensor(u), torch.tensor(v)
    g = dgl.graph((u, v))

    symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
               'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
               'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
               'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    feats = []
    for atom in mol.GetAtoms():
        f0 = one_hot(atom.GetSymbol(), symbols)
        f1 = one_hot(atom.GetFormalCharge(), [1, 2, 3, 4, 5, 6, 7, 8])
        f2 = one_hot(atom.GetDegree(), [1, 2, 3, 4, 5, 6, 7, 8])
        f3 = torch.tensor([float(atom.GetIsAromatic())])
        f4 = torch.tensor([float(atom.IsInRing())])
        feats.append(torch.cat([f0, f1, f2, f3, f4]))

    feats_1 = torch.stack(feats, 0)
    nums_n = min(max(max(u), max(v)) + 1, nums_atom)
    feats_2 = feats_1[:nums_n]
    bg = dgl.to_bidirected(g)
    bg.ndata['feat'] = feats_2
    return bg


def _make_protein_data(seq_len=100):
    """Create synthetic protein data (sequence embedding + contact graph)."""
    T = torch.randn(1, seq_len, 30)

    u_list, v_list, weights = [], [], []
    for i in range(seq_len):
        if i > 0:
            u_list.extend([i, i - 1])
            v_list.extend([i - 1, i])
            weights.extend([1.0, 1.0])
    tg = dgl.graph((u_list, v_list), num_nodes=seq_len)
    tg.ndata['feat'] = torch.randn(seq_len, 30)
    tg.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return T, tg


@pytest.fixture
def real_drug_smiles(biosnap_dir):
    """Load the first drug SMILES from the official BIOSNAP dataset."""
    smi_path = os.path.join(biosnap_dir, "drug_smi_raw.pkl")
    if not os.path.exists(smi_path):
        pytest.skip("BIOSNAP drug_smi_raw.pkl not found")
    with open(smi_path, 'rb') as f:
        drugs = pickle.load(f)
    return list(drugs.values())[0]


CHECKPOINT_NAMES = [
    "BIOSNAP_CV1.pth",
    "BIOSNAP_unseen_D.pth",
    "BIOSNAP_unseen_T.pth",
]


# ════════════════════════════════════════════════════════════
#  Basic forward pass (random weights)
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestForwardPassBasic:
    """Test forward pass with random weights (no checkpoint needed)."""

    def test_synthetic_forward_pass(self, build_model):
        model = build_model()
        model.eval()

        drug_g = _smi_2_graph("CCO")  # ethanol
        T, tg = _make_protein_data(seq_len=50)
        D_G = dgl.batch([drug_g])
        T_G = dgl.batch([tg])

        with torch.no_grad():
            output = model((D_G, T, T_G))

        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_batch_forward_pass(self, build_model):
        """Test with batch_size > 1."""
        model = build_model()
        model.eval()

        seq_len = 50
        batch_size = 3
        drug_graphs = [_smi_2_graph(smi) for smi in ["CCO", "c1ccccc1", "CC(=O)O"]]
        T = torch.randn(batch_size, seq_len, 30)
        protein_graphs = [_make_protein_data(seq_len)[1] for _ in range(batch_size)]

        D_G = dgl.batch(drug_graphs)
        T_G = dgl.batch(protein_graphs)

        with torch.no_grad():
            output = model((D_G, T, T_G))

        assert output.shape == (batch_size, 2)
        assert torch.isfinite(output).all()


# ════════════════════════════════════════════════════════════
#  Forward pass with real BIOSNAP data + official checkpoints
# ════════════════════════════════════════════════════════════

@pytest.mark.checkpoint
class TestForwardPassWithCheckpoint:
    """Forward pass using real SMILES and official pretrained weights."""

    @pytest.mark.parametrize("ckpt_name", CHECKPOINT_NAMES)
    def test_checkpoint_forward_pass(self, build_model, real_drug_smiles, ckpt_name, checkpoint_dir):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        model = build_model()
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        drug_g = _smi_2_graph(real_drug_smiles)
        T, tg = _make_protein_data(seq_len=100)
        D_G = dgl.batch([drug_g])
        T_G = dgl.batch([tg])

        with torch.no_grad():
            output = model((D_G, T, T_G))

        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
        assert torch.isfinite(output).all(), f"{ckpt_name}: Output contains non-finite values"
        assert not torch.isnan(output).any(), f"{ckpt_name}: Output contains NaN"
        assert not torch.isinf(output).any(), f"{ckpt_name}: Output contains Inf"

    @pytest.mark.parametrize("ckpt_name", CHECKPOINT_NAMES)
    def test_checkpoint_produces_probabilities(self, build_model, real_drug_smiles, ckpt_name, checkpoint_dir):
        """After softmax, outputs should be valid probabilities."""
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        model = build_model()
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        drug_g = _smi_2_graph(real_drug_smiles)
        T, tg = _make_protein_data(seq_len=100)
        D_G = dgl.batch([drug_g])
        T_G = dgl.batch([tg])

        with torch.no_grad():
            output = model((D_G, T, T_G))
            probs = torch.softmax(output, dim=1)

        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of range"
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5), (
            "Probabilities don't sum to 1"
        )


# ════════════════════════════════════════════════════════════
#  Multiple real SMILES forward passes
# ════════════════════════════════════════════════════════════

@pytest.mark.checkpoint
class TestMultipleDrugForwardPasses:

    def test_five_drugs_forward(self, build_model, biosnap_dir, checkpoint_dir):
        """Process 5 real drugs from BIOSNAP through the model."""
        smi_path = os.path.join(biosnap_dir, "drug_smi_raw.pkl")
        ckpt_path = os.path.join(checkpoint_dir, "BIOSNAP_CV1.pth")
        if not os.path.exists(smi_path):
            pytest.skip("BIOSNAP data not found")
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not found")

        with open(smi_path, 'rb') as f:
            drugs = pickle.load(f)

        model = build_model()
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        success = 0
        for drug_id, smi in list(drugs.items())[:5]:
            try:
                drug_g = _smi_2_graph(smi)
                T, tg = _make_protein_data(seq_len=100)
                D_G = dgl.batch([drug_g])
                T_G = dgl.batch([tg])

                with torch.no_grad():
                    output = model((D_G, T, T_G))

                assert output.shape == (1, 2)
                assert torch.isfinite(output).all()
                success += 1
            except Exception as e:
                print(f"Warning: Drug {drug_id} ({smi}) failed: {e}")

        assert success >= 4, f"Only {success}/5 drugs succeeded"
