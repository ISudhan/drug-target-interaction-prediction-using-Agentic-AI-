"""
GNNBlock-based drug encoding module for GNNBlockDTI.

Architecture from paper (Deng et al., 2025):
  Drug molecular graph → GNNBlock₀ (initial) → [GNNBlock₁...GNNBlockₗ] with Gating Units → AvgPooling

Each GNNBlock contains:
  - N-1 GAT layers (Graph Attention Network) for substructure capture
  - 1 GCN layer (Graph Convolution) for feature enhancement
  - GLU (Gated Linear Unit) on the GCN output
  - Residual connection (add GAT output + GLU(GCN output)) * scale
  - LayerNorm
  - Gating Unit (GRU-like) between blocks

Checkpoint naming (must match for strict=True loading):
  net_D.gcn_1       → initial GATGCN_Block
  net_D.gcn_2s.{i}  → subsequent GATGCN_Blocks
  net_D.LNs.{i}     → LayerNorms
  net_D.gate         → Gated_NN
"""
import torch
from torch import nn
import dgl
from dgl.nn.pytorch import GATConv, GraphConv, AvgPooling


class GATGCN_Block(nn.Module):
    """A single GNNBlock combining GAT layers + GCN feature enhancement.

    Architecture (from paper Figure 1b):
        input → GAT_layer¹ → ... → GAT_layer^(n-1) → GCN_layer^(n) → output
        Returns both the GAT output (h) and the GCN-enhanced output (h_1).

    Paper: "GNNBlock incorporates multiple layers of GNNs to capture hidden
    structural patterns within local ranges."

    The last GNN layer (GCN) serves as "feature enhancement" — re-encoding
    structural features to double the dimension for GLU gating.

    Args:
        input_size: Input node feature dimension
        hidden_size: Hidden dimension for GAT layers (total across all heads)
        output_size: Output dimension of GCN enhancement layer (typically 2× hidden)
        nums_layer: Total number of GNN layers (GAT layers = nums_layer-1, GCN layers = 1)
        n_heads: Number of attention heads in GAT layers (default: 4)
    """
    def __init__(self, input_size, hidden_size, output_size, nums_layer, n_heads=4):
        super(GATGCN_Block, self).__init__()
        assert nums_layer > 1, "GNNBlock requires at least 2 layers (1 GAT + 1 GCN)"

        # GAT layers for substructure feature capture
        # First GAT layer: input_size → hidden_size (via n_heads × hidden_size//n_heads)
        netlist = [GATConv(input_size, hidden_size // n_heads, n_heads,
                           activation=nn.ReLU(), allow_zero_in_degree=True)]
        # Additional GAT layers: hidden_size → hidden_size
        for i in range(nums_layer - 2):
            netlist.append(GATConv(hidden_size, hidden_size // n_heads, n_heads,
                                   activation=nn.ReLU(), allow_zero_in_degree=True))
        self.net = nn.ModuleList(netlist)

        # GCN layer for feature enhancement (last layer in the block)
        # Paper: "feature enhancement strategy to re-encode structural features"
        self.net_1 = GraphConv(hidden_size, output_size,
                               norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)

    def forward(self, g, feats):
        """Forward pass through GAT layers then GCN enhancement.

        Args:
            g: DGL graph
            feats: Node features [num_nodes, input_size]

        Returns:
            feats: GAT output [num_nodes, hidden_size]
            feats_1: GCN-enhanced output [num_nodes, output_size]
        """
        for layer in self.net:
            feats = layer(g, feats)
            # GATConv returns [num_nodes, n_heads, out_dim], reshape to [num_nodes, hidden_size]
            feats = feats.reshape(feats.shape[0], -1)
        feats_1 = self.net_1(g, feats)

        return feats, feats_1


class Gated_NN(nn.Module):
    """Gating Unit (GU) between GNNBlocks.

    Paper (Figure 1c): GRU-like gating mechanism that filters redundant information
    and coordinates substructural features across blocks.

    Equations (from paper):
        R = σ(h⁰ · W_r1 + h¹ · W_r2 + b_r)         (Reset gate)
        Z = σ(h⁰ · W_z1 + h¹ · W_z2 + b_z)         (Update gate)
        h² = tanh(h¹ · W_h1 + (R ⊙ h⁰) · W_h2 + b_h)  (Candidate)
        h = Z ⊙ h⁰ + (1-Z) ⊙ h²                    (Output)

    where h⁰ is the previous block output and h¹ is the current block output.

    Args:
        hidden_size: Dimension of node features
    """
    def __init__(self, hidden_size):
        super(Gated_NN, self).__init__()
        self.hidden_size = hidden_size
        w_r1, w_r2, b_r = self.get_params()
        w_z1, w_z2, b_z = self.get_params()
        w_h1, w_h2, b_h = self.get_params()

        self.w_r1 = nn.Parameter(w_r1)
        self.w_r2 = nn.Parameter(w_r2)
        self.b_r = nn.Parameter(b_r)

        self.w_z1 = nn.Parameter(w_z1)
        self.w_z2 = nn.Parameter(w_z2)
        self.b_z = nn.Parameter(b_z)

        self.w_h1 = nn.Parameter(w_h1)
        self.w_h2 = nn.Parameter(w_h2)
        self.b_h = nn.Parameter(b_h)

    def forward(self, h0, h1):
        """Apply gating between previous block output (h0) and current block output (h1).

        Args:
            h0: Previous block output [num_nodes, hidden_size]
            h1: Current block output [num_nodes, hidden_size]

        Returns:
            h: Gated output [num_nodes, hidden_size]
        """
        R = torch.sigmoid(h0 @ self.w_r1 + h1 @ self.w_r2 + self.b_r)
        Z = torch.sigmoid(h0 @ self.w_z1 + h1 @ self.w_z2 + self.b_z)
        h2 = torch.tanh(h1 @ self.w_h1 + ((R * h0) @ self.w_h2) + self.b_h)
        h = Z * h0 + (1 - Z) * h2

        return h

    def get_params(self):
        return (self.normal((self.hidden_size, self.hidden_size)),
                self.normal((self.hidden_size, self.hidden_size)),
                torch.zeros(self.hidden_size))

    def normal(self, shape):
        return torch.normal(0, 1, shape) * 0.01


class GNNBlocks(nn.Module):
    """Complete drug encoding module using stacked GNNBlocks with Gating Units.

    Architecture (paper Figure 1a, drug branch):
        GNNBlock₀(initial) → GNNBlock₁ → GU → ... → GNNBlockₗ → GU → AvgPooling

    Each subsequent GNNBlock (1..L):
        1. GATGCN_Block produces (h_gat, h_gcn_enhanced)
        2. Residual: h_combined = (h_gat + dropout(GLU(h_gcn_enhanced))) × scale
        3. LayerNorm
        4. Gating Unit with previous block output

    Module naming matches checkpoint (critical for strict=True loading):
        gcn_1   → initial GATGCN_Block
        gcn_2s  → ModuleList of subsequent GATGCN_Blocks
        LNs     → ModuleList of LayerNorms
        gate    → Gated_NN

    Args:
        input_size: Atom feature dimension (64)
        hidden_size: Hidden dimension (96)
        n_layer: Number of subsequent GNNBlocks (5)
        n_gnn: Number of GNN layers per subsequent block (2)
        dropout: Dropout rate (0.2)
    """
    def __init__(self, input_size, hidden_size, n_layer, n_gnn, dropout):
        super(GNNBlocks, self).__init__()
        self.n_layer = n_layer

        # Initial GNNBlock: input_size → hidden_size → hidden_size*2
        # Paper: GNNBlock₀ with subscript 3 (3 GNN layers: 2 GAT + 1 GCN)
        # BUG FIX: Official code uses undefined 'GCN_Block' — checkpoint proves
        # this is GATGCN_Block (contains GAT attention params attn_l, attn_r)
        # Checkpoint key: net_D.gcn_1
        self.gcn_1 = GATGCN_Block(input_size, hidden_size, hidden_size * 2, nums_layer=3)

        # Subsequent GNNBlocks (L blocks)
        # Checkpoint key: net_D.gcn_2s.{0..n_layer-1}
        gcn_2s = []
        for _ in range(n_layer):
            block = GATGCN_Block(hidden_size * 2, hidden_size * 2,
                                 hidden_size * 4, nums_layer=n_gnn)
            gcn_2s.append(block)
        self.gcn_2s = nn.ModuleList(gcn_2s)

        # LayerNorm for each subsequent block
        LNs = []
        for _ in range(n_layer):
            ln = nn.LayerNorm(hidden_size * 2)
            LNs.append(ln)
        self.LNs = nn.ModuleList(LNs)

        # Gating Unit (shared across all block transitions)
        self.gate = Gated_NN(hidden_size * 2)

        # Scale factor for residual connection: 1/√2
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))

        # GLU for feature enhancement output (halves dimension: 4*hidden → 2*hidden)
        self.glu = nn.GLU(dim=-1)

        self.do = nn.Dropout(dropout)

        # PAPER-vs-CODE-DISCREPANCY: Paper says "max-pooling layer" in text and
        # "Maxpool" in diagram. However, official code (models.py L151) uses
        # AvgPooling() despite naming the variable 'self.maxpool'. Since the
        # checkpoint doesn't contain pooling parameters (both are parameter-free),
        # we match the official code to reproduce trained model behavior.
        self.maxpool = AvgPooling()

    def forward(self, g, feats):
        """Encode drug molecular graph through stacked GNNBlocks.

        Args:
            g: DGL molecular graph (batched)
            feats: Atom features [num_nodes, input_size]

        Returns:
            output: Drug embedding [batch_size, hidden_size*2]
        """
        # Initial block
        h, h_1 = self.gcn_1(g, feats)

        # Subsequent blocks with gating
        for i in range(self.n_layer):
            block = self.gcn_2s[i]
            LN = self.LNs[i]

            # GATGCN_Block: returns (GAT_output, GCN_enhanced_output)
            h1, h1_1 = block(g, h_1)

            # Residual: GAT_output + dropout(GLU(GCN_enhanced_output))
            # GLU halves the dimension: hidden*4 → hidden*2
            h2 = (h1 + self.do(self.glu(h1_1))) * self.scale.to(feats.device)

            # LayerNorm
            h2 = LN(h2)

            # Gating Unit: filter redundant information
            h_1 = self.gate(h_1, h2)

        # Graph-level readout via pooling
        output = self.maxpool(g, h_1)

        return output
