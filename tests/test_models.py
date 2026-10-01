"""Test structural properties of the GNNBlockDTI model architecture.

Verifies module existence, layer counts, dimensions, and output shapes
against the official checkpoint structure and paper specification.

All tests are 'unit' tests — they do not require pretrained models or data.
"""
import pytest
import torch
import torch.nn as nn
import dgl

from src.models.gnn_blocks import GNNBlocks, GATGCN_Block, Gated_NN
from src.models.protein import WGCN, MultiscaleCNN, FeatureFusion
from src.models.model import GNNBlockDTI
from dgl.nn.pytorch import GATConv, GraphConv, AvgPooling


# ════════════════════════════════════════════════════════════
#  Model construction
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGNNBlockDTIStructure:
    """Verify top-level GNNBlockDTI has expected sub-modules."""

    def test_has_drug_network(self, build_model):
        model = build_model()
        assert hasattr(model, 'net_D'), "Missing drug encoding network"
        assert isinstance(model.net_D, GNNBlocks)

    def test_has_protein_sequence_network(self, build_model):
        model = build_model()
        assert hasattr(model, 'net_T'), "Missing protein sequence network"
        assert isinstance(model.net_T, MultiscaleCNN)

    def test_has_protein_graph_network(self, build_model):
        model = build_model()
        assert hasattr(model, 'net_T1'), "Missing protein graph network"
        assert isinstance(model.net_T1, WGCN)

    def test_has_feature_fusion(self, build_model):
        model = build_model()
        assert hasattr(model, 'FF'), "Missing FeatureFusion module"
        assert isinstance(model.FF, (FeatureFusion))

    def test_has_pair_net(self, build_model):
        model = build_model()
        assert hasattr(model, 'pair_net'), "Missing pair_net classifier MLP"
        assert isinstance(model.pair_net, nn.Sequential)

    def test_has_final_classifier(self, build_model):
        model = build_model()
        assert hasattr(model, 'fc'), "Missing final classifier layer"
        assert isinstance(model.fc, nn.Linear)

    def test_classifier_dimensions(self, build_model):
        """fc: Linear(512, 2) — verified against checkpoint key fc.weight shape [2, 512]."""
        model = build_model()
        assert model.fc.in_features == 512
        assert model.fc.out_features == 2

    def test_total_parameter_count(self, build_model):
        """Official checkpoints have exactly 3,342,978 parameters."""
        model = build_model()
        total = sum(p.numel() for p in model.parameters())
        assert total == 3_342_978, f"Parameter count {total} != expected 3,342,978"

    def test_state_dict_key_count(self, build_model):
        """All 3 checkpoints have exactly 97 keys."""
        model = build_model()
        assert len(model.state_dict()) == 97


# ════════════════════════════════════════════════════════════
#  Drug Branch: GNNBlocks
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGNNBlocksStructure:
    """Verify GNNBlocks internal structure matches checkpoint and paper."""

    def test_initial_block_exists(self, build_model):
        model = build_model()
        assert hasattr(model.net_D, 'gcn_1'), "Missing initial GNNBlock (gcn_1)"

    def test_initial_block_is_gatgcn(self, build_model):
        """Checkpoint proves gcn_1 contains GAT params (attn_l, attn_r)."""
        model = build_model()
        assert isinstance(model.net_D.gcn_1, GATGCN_Block)

    def test_initial_block_has_3_layers(self, build_model):
        """GNNBlock₀: 2 GAT layers (net.0, net.1) + 1 GCN layer (net_1) = 3 total.
        Verified by checkpoint keys: net_D.gcn_1.net.0.*, net_D.gcn_1.net.1.*, net_D.gcn_1.net_1.*
        """
        model = build_model()
        block0 = model.net_D.gcn_1
        assert len(block0.net) == 2, f"Expected 2 GAT layers in initial block, got {len(block0.net)}"
        assert hasattr(block0, 'net_1'), "Missing GCN enhancement layer in initial block"

    def test_initial_block_gat_types(self, build_model):
        model = build_model()
        block0 = model.net_D.gcn_1
        for i, layer in enumerate(block0.net):
            assert isinstance(layer, GATConv), f"GAT layer {i} in initial block is not GATConv"

    def test_initial_block_gcn_type(self, build_model):
        model = build_model()
        assert isinstance(model.net_D.gcn_1.net_1, GraphConv)

    def test_subsequent_block_count(self, build_model):
        """n_layer=5 → 5 subsequent GATGCN_Blocks."""
        model = build_model()
        assert hasattr(model.net_D, 'gcn_2s')
        assert len(model.net_D.gcn_2s) == 5

    def test_subsequent_blocks_have_2_layers(self, build_model):
        """Each subsequent block: 1 GAT layer + 1 GCN layer = n_gnn=2.
        Checkpoint keys: net_D.gcn_2s.{i}.net.0.* (1 GAT) + net_D.gcn_2s.{i}.net_1.* (1 GCN).
        """
        model = build_model()
        for i, block in enumerate(model.net_D.gcn_2s):
            assert len(block.net) == 1, (
                f"Subsequent block {i}: expected 1 GAT layer, got {len(block.net)}"
            )
            assert hasattr(block, 'net_1'), f"Subsequent block {i}: missing GCN layer"

    def test_layernorm_count(self, build_model):
        """One LayerNorm per subsequent block."""
        model = build_model()
        assert len(model.net_D.LNs) == 5

    def test_layernorm_dimension(self, build_model):
        """LayerNorm dimension = hidden_size * 2 = 192."""
        model = build_model()
        for ln in model.net_D.LNs:
            assert ln.normalized_shape == (192,)

    def test_gate_exists(self, build_model):
        model = build_model()
        assert hasattr(model.net_D, 'gate')
        assert isinstance(model.net_D.gate, Gated_NN)

    def test_gate_dimensions(self, build_model):
        """Gate hidden_size = hidden_size * 2 = 192."""
        model = build_model()
        assert model.net_D.gate.hidden_size == 192

    def test_pooling_is_avg(self, build_model):
        """Official code uses AvgPooling despite naming it 'maxpool'."""
        model = build_model()
        assert isinstance(model.net_D.maxpool, AvgPooling), (
            f"Expected AvgPooling (matching official code), got {type(model.net_D.maxpool)}"
        )


# ════════════════════════════════════════════════════════════
#  Protein Sequence Branch: MultiscaleCNN
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMultiscaleCNNStructure:

    def test_input_embedding(self, build_model):
        model = build_model()
        cnn = model.net_T
        assert cnn.embed.in_features == 30, "ProtBERT logit dim should be 30"
        assert cnn.embed.out_features == 128

    def test_cnn_kernel_sizes(self, build_model):
        """Paper: kernel sizes 5, 7, 13."""
        model = build_model()
        cnn = model.net_T
        assert cnn.cnn1.kernel_size == (5,)
        assert cnn.cnn2.kernel_size == (7,)
        assert cnn.cnn3.kernel_size == (13,)

    def test_cnn_channels(self, build_model):
        """Conv channels: 32, 64, 96."""
        model = build_model()
        cnn = model.net_T
        assert cnn.cnn1.out_channels == 32
        assert cnn.cnn2.out_channels == 64
        assert cnn.cnn3.out_channels == 96

    def test_fc_dimensions(self, build_model):
        """FC layers match conv channels."""
        model = build_model()
        cnn = model.net_T
        assert cnn.fc1.in_features == 32 and cnn.fc1.out_features == 32
        assert cnn.fc2.in_features == 64 and cnn.fc2.out_features == 64
        assert cnn.fc3.in_features == 96 and cnn.fc3.out_features == 96

    def test_output_dimension(self):
        """Output = cat(fc1, fc2, fc3) along feature dim = 32+64+96 = 192."""
        cnn = MultiscaleCNN(inputs=30, embed_dim=128, nums_conv=[32, 64, 96],
                            ksize=[5, 7, 13], dropout=0.2)
        x = torch.randn(2, 100, 30)
        out = cnn(x)
        assert out.shape == (2, 100, 192), f"Expected (2, 100, 192), got {out.shape}"


# ════════════════════════════════════════════════════════════
#  Protein Graph Branch: WGCN
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestWGCNStructure:

    def test_has_three_gcn_layers(self, build_model):
        model = build_model()
        wgcn = model.net_T1
        assert isinstance(wgcn.gcn1, GraphConv)
        assert isinstance(wgcn.gcn2, GraphConv)
        assert isinstance(wgcn.gcn3, GraphConv)

    def test_gcn_dimensions(self, build_model):
        """gcn1: 30→128, gcn2: 128→128, gcn3: 128→256."""
        model = build_model()
        wgcn = model.net_T1
        # GraphConv stores weights as (in, out) — checkpoint shows e.g. gcn1.weight: [30, 128]
        assert wgcn.LN1.normalized_shape == (128,)
        assert wgcn.LN2.normalized_shape == (128,)
        assert wgcn.LN3.normalized_shape == (256,)

    def test_has_layernorms(self, build_model):
        model = build_model()
        wgcn = model.net_T1
        assert hasattr(wgcn, 'LN1')
        assert hasattr(wgcn, 'LN2')
        assert hasattr(wgcn, 'LN3')


# ════════════════════════════════════════════════════════════
#  Feature Fusion
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFeatureFusionStructure:

    def test_fusion_dimensions(self, build_model):
        """FF.fc1: (192, 384), FF.fc2: (256, 384) — from checkpoint."""
        model = build_model()
        ff = model.FF
        assert ff.fc1.in_features == 192
        assert ff.fc1.out_features == 384
        assert ff.fc2.in_features == 256
        assert ff.fc2.out_features == 384

    def test_fusion_output_shape(self):
        ff = FeatureFusion(size1=192, size2=256, hid_size=384)
        x1 = torch.randn(2, 100, 192)
        x2 = torch.randn(2, 100, 256)
        out = ff(x1, x2)
        assert out.shape == (2, 384), f"Expected (2, 384), got {out.shape}"


# ════════════════════════════════════════════════════════════
#  Pair Network / MLP
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPairNetStructure:

    def test_pair_net_dimensions(self, build_model):
        """pair_net: 576→1024→1024→512 — from checkpoint keys."""
        model = build_model()
        layers = list(model.pair_net)
        # Layer 0: Linear(576, 1024)
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 576  # drug_len(192) + hid_size(384)
        assert layers[0].out_features == 1024
        # Layer 1: ReLU
        assert isinstance(layers[1], nn.ReLU)
        # Layer 2: Dropout
        assert isinstance(layers[2], nn.Dropout)
        # Layer 3: Linear(1024, 1024)
        assert isinstance(layers[3], nn.Linear)
        assert layers[3].in_features == 1024
        assert layers[3].out_features == 1024
        # Layer 6: Linear(1024, 512)
        assert isinstance(layers[6], nn.Linear)
        assert layers[6].in_features == 1024
        assert layers[6].out_features == 512


# ════════════════════════════════════════════════════════════
#  Output shape
# ════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestModelOutputShape:

    def test_forward_output_shape(self, build_model):
        """Forward pass produces [batch, 2] output."""
        model = build_model()
        model.eval()

        batch_size = 2
        num_nodes_drug = 10
        num_nodes_prot = 50
        seq_len = num_nodes_prot

        # Drug graph
        src_d = torch.randint(0, num_nodes_drug, (20,))
        dst_d = torch.randint(0, num_nodes_drug, (20,))
        g_d = dgl.graph((src_d, dst_d), num_nodes=num_nodes_drug)
        g_d.ndata['feat'] = torch.randn(num_nodes_drug, 64)
        D_G = dgl.batch([g_d] * batch_size)

        # Protein sequence
        T = torch.randn(batch_size, seq_len, 30)

        # Protein graph
        u_list, v_list, weights = [], [], []
        for i in range(num_nodes_prot):
            if i > 0:
                u_list.extend([i, i - 1])
                v_list.extend([i - 1, i])
                weights.extend([1.0, 1.0])
        g_t = dgl.graph((u_list, v_list), num_nodes=num_nodes_prot)
        g_t.ndata['feat'] = torch.randn(num_nodes_prot, 30)
        g_t.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
        T_G = dgl.batch([g_t] * batch_size)

        with torch.no_grad():
            output = model((D_G, T, T_G))

        assert output.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"
