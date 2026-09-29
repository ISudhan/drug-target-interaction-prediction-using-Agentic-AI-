import torch
from torch import nn

from .gnn_blocks import GNNBlocks
from .protein import WGCN, MultiscaleCNN, ConcatenationFusion, BilinearFusion, GatedFusion

class GNNBlockDTI(nn.Module):
    """GNNBlockDTI: Efficient substructure feature encoding based on graph 
    neural network blocks for drug-target interaction prediction.

    Architecture (from paper):
        Drug branch (GNNBlocks): Encodes drug molecular graph → drug embedding
        Protein branch (MultiscaleCNN + WGCN): Encodes sequence and contact map → protein embedding
        Classifier: MLP concatenating drug and protein embeddings → DT interaction prediction
    """
    def __init__(self, net_D, net_T, net_T1, Drug_len, Target_len, Target_len1, hid_size, dropout=0.2, fusion_type='concat'):
        """
        Args:
            net_D: Drug encoding module (GNNBlocks)
            net_T: Protein sequence encoding module (MultiscaleCNN)
            net_T1: Protein graph encoding module (WGCN)
            Drug_len: Drug embedding dimension (192)
            Target_len: Output dim of MultiscaleCNN (192)
            Target_len1: Output dim of WGCN (256)
            hid_size: Fused protein representation dimension (384)
            dropout: Dropout rate (0.2)
            fusion_type: Type of fusion to use ('concat', 'bilinear', 'gated')
        """
        super(GNNBlockDTI, self).__init__()
        self.net_D = net_D
        self.net_T = net_T
        self.net_T1 = net_T1
        
        if fusion_type == 'concat':
            self.FF = ConcatenationFusion(Target_len, Target_len1, hid_size)
            protein_rep_size = hid_size * 2
        elif fusion_type == 'bilinear':
            self.FF = BilinearFusion(Target_len, Target_len1, hid_size)
            protein_rep_size = hid_size
        elif fusion_type == 'gated':
            self.FF = GatedFusion(Target_len, Target_len1, hid_size)
            protein_rep_size = hid_size
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
            
        # Classifier Network
        self.pair_net = nn.Sequential(
            nn.Linear(Drug_len + protein_rep_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, pairs):
        """
        Args:
            pairs: Tuple containing (D_G, T, T_G)
                D_G: Batched drug molecular graphs (DGL)
                T: Batched protein sequence features [batch, seq_len, 30]
                T_G: Batched protein contact graphs (DGL)
        
        Returns:
            output: Prediction logits [batch, 2]
        """
        D_G, T, T_G = pairs
        
        # Extract node features and edge weights
        Dfeats = D_G.ndata['feat']
        Tfeats = T_G.ndata['feat']
        Tweights = T_G.edata["weight"]

        # Drug encoding
        D_x = self.net_D(D_G, Dfeats)
        
        # Protein encoding (graph and sequence)
        T_x1 = self.net_T1(T_G, Tfeats, Tweights)
        T_x = self.net_T(T)
        
        # Reshape WGCN output to match sequence structure
        # [total_nodes, dim] -> [batch, seq_len, dim]
        batch_num_nodes = T_G.batch_num_nodes().tolist()
        T_x1_split = torch.split(T_x1, batch_num_nodes)
        max_len = T_x.shape[1]
        T_x1_padded = torch.zeros(len(batch_num_nodes), max_len, T_x1.shape[-1], device=T_x1.device)
        for i, t in enumerate(T_x1_split):
            T_x1_padded[i, :t.shape[0], :] = t
        T_x1 = T_x1_padded
        
        # Feature fusion
        T_xco = self.FF(T_x, T_x1)
        
        # Concatenate drug and protein features
        DT = torch.cat([D_x, T_xco], dim=-1)
        
        # Classifier
        DT_x = self.pair_net(DT)
        output = self.fc(DT_x)

        # Output shape is [batch, 2] since self.fc is Linear(512, 2)
        # However, the official code uses output.squeeze(-1) which does nothing to [batch, 2]
        return output.squeeze(-1)
