import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, MaxPooling


class WGCN(nn.Module):
    """Weighted Graph Convolutional Network for protein contact map encoding.

    Architecture:
        Graph with weighted edges (contact probabilities)
        → GCN (input_size → hidden_size) → LayerNorm
        → GCN (hidden_size → hidden_size) → LayerNorm
        → GCN (hidden_size → hidden_size*2) → LayerNorm
        Output: Node embeddings [num_nodes, hidden_size*2]

    Args:
        input_size: ProtBERT feature dimension per residue (30)
        hidden_size: Hidden dimension (128)
        dropout: Dropout rate (0.2)
    """
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(WGCN, self).__init__()
        self.norm = EdgeWeightNorm(norm='both')

        self.gcn1 = GraphConv(input_size, hidden_size,
                              norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        self.gcn2 = GraphConv(hidden_size, hidden_size,
                              norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        self.gcn3 = GraphConv(hidden_size, hidden_size * 2,
                              norm='both', activation=nn.ReLU(), allow_zero_in_degree=True)
        self.LN1 = nn.LayerNorm(hidden_size)
        self.LN2 = nn.LayerNorm(hidden_size)
        self.LN3 = nn.LayerNorm(hidden_size * 2)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.glu = nn.GLU(dim=-1)
        self.do = nn.Dropout(dropout)
        self.maxpool = MaxPooling()

    def forward(self, g, feats, edge_weight):
        """
        Args:
            g: DGL graph
            feats: Protein residue features [num_nodes, input_size]
            edge_weight: Contact probabilities for edges
        """
        norm_edge_weight = self.norm(g, edge_weight)
        h1 = self.gcn1(g, feats, edge_weight=norm_edge_weight)
        h1 = self.LN1(h1)
        h2 = self.gcn2(g, h1, edge_weight=norm_edge_weight)
        h2 = self.LN2(h2)
        h3 = self.gcn3(g, h2, edge_weight=norm_edge_weight)
        h3 = self.LN3(h3)

        return h3


class MultiscaleCNN(nn.Module):
    """Multi-scale 1D Convolutional Neural Network for protein sequence encoding.

    Architecture:
        Sequence embeddings → Linear(input → embed_dim)
        → Conv1D branch 1 → ReLU → FC1
        → Conv1D branch 2 → ReLU → FC2
        → Conv1D branch 3 → ReLU → FC3
        → Concat branches along feature dimension

    Args:
        inputs: Input dimension (ProtBERT logits, 30)
        embed_dim: Initial embedding projection dimension (128)
        nums_conv: List of output channels for the 3 branches (e.g. [32, 64, 96])
        ksize: List of kernel sizes for the 3 branches (e.g. [5, 7, 13])
        dropout: Dropout rate
    """
    def __init__(self, inputs, embed_dim, nums_conv, ksize, dropout, **kwargs):
        super(MultiscaleCNN, self).__init__(**kwargs)
        self.embed = nn.Linear(inputs, embed_dim)
        self.cnn1 = nn.Conv1d(embed_dim, nums_conv[0], kernel_size=ksize[0], padding="same")
        self.cnn2 = nn.Conv1d(nums_conv[0], nums_conv[1], kernel_size=ksize[1], padding="same")
        self.cnn3 = nn.Conv1d(nums_conv[1], nums_conv[2], kernel_size=ksize[2], padding="same")
        self.fc1 = nn.Linear(nums_conv[0], nums_conv[0])
        self.fc2 = nn.Linear(nums_conv[1], nums_conv[1])
        self.fc3 = nn.Linear(nums_conv[2], nums_conv[2])
        self.relu = nn.ReLU()

    def forward(self, X):
        """
        Args:
            X: Sequence features [batch_size, seq_len, inputs]
        """
        X = self.embed(X)
        X = X.permute(0, 2, 1)  # [batch, embed_dim, seq_len] for Conv1d
        
        # Branch 1
        X1 = self.cnn1(X)
        X1 = self.relu(X1)
        
        # Branch 2 (sequential over Branch 1 output in official code, not parallel)
        # Note: Code actually chains them sequentially X -> X1 -> X2 -> X3, 
        # but the paper diagram shows a cascade/parallel structure. We follow code.
        X2 = self.cnn2(X1)
        X2 = self.relu(X2)
        
        # Branch 3
        X3 = self.cnn3(X2)
        X3 = self.relu(X3)

        # Map back and pass through FC
        X1 = self.fc1(X1.permute(0, 2, 1))
        X2 = self.fc2(X2.permute(0, 2, 1))
        X3 = self.fc3(X3.permute(0, 2, 1))
        
        # Concatenate features
        X_co = torch.cat((X1, X2, X3), dim=-1)

        return X_co


class FeatureFusion(nn.Module):
    """Protein Feature Fusion module.

    Fuses CNN sequential features and WGCN spatial features.

    Architecture:
        x1 (CNN) → Linear(size1 → hid_size) → L2-norm
        x2 (WGCN) → Linear(size2 → hid_size) → L2-norm
        x_co = x1 + x2
        AdaptiveMaxPool1d → final protein representation
    """
    def __init__(self, size1, size2, hid_size):
        super(FeatureFusion, self).__init__()
        self.fc1 = nn.Linear(size1, hid_size)
        self.fc2 = nn.Linear(size2, hid_size)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x1, x2):
        """
        Args:
            x1: CNN sequence features [batch, seq_len, size1]
            x2: WGCN graph features [batch, num_nodes, size2] 
                Note: num_nodes usually matches seq_len
        """
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        
        # Element-wise addition
        xco = x1 + x2
        
        # Global max pooling over sequence length
        # Permute to [batch, hid_size, seq_len] for AdaptiveMaxPool1d
        op = self.maxpool(xco.permute(0, 2, 1)).squeeze(-1)

        return op
