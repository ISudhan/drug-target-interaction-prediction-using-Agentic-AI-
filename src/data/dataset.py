import dgl
import torch
from torch.utils.data import Dataset


class DTIDataset(Dataset):
    """Dataset for Drug-Target Interaction prediction.
    
    Loads and filters interaction pairs, matching them with their corresponding
    drug and target representations.
    
    Args:
        data: List of [drug_id, target_id, label] pairs
        drug_graph: Dict mapping drug_id to DGL graph
        target_embedding: Dict mapping target_id to padded sequence features
        target_graph: Dict mapping target_id to DGL contact map graph
    """
    def __init__(self, data, drug_graph, target_embedding, target_graph):
        super(DTIDataset, self).__init__()
        D, T, Tg, Y = [], [], [], []
        
        # Iterate through pairs and collect corresponding data
        for ent in data:
            drug_id = str(ent[0])
            prot_id = str(ent[1])
            label = int(ent[2])
            
            try:
                # Bug fix: Official code used undefined `drug_data` instead of `drug_graph`
                # Bug fix: Official code used undefined `prot_seq`, `prot_data` instead of embedding/graph
                D.append(drug_graph[drug_id])
                T.append(target_embedding[prot_id])
                Tg.append(target_graph[prot_id])
                Y.append(label)
            except KeyError:
                # Skip pairs where we lack features (e.g. processing failed)
                pass
                
        self.D = D
        self.T = T
        self.Tg = Tg
        self.Y = Y
        
    def __getitem__(self, idx):
        """Returns tuple of (drug_graph, target_seq, target_graph, label)."""
        return self.D[idx], self.T[idx], self.Tg[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.Y)


def collate_fn(batch):
    """Collate function for DTI DataLoader.
    
    Batches DGL graphs together and stacks sequence tensors.
    """
    D_graphs, T0, T_graphs, Y0 = map(list, zip(*batch))
    
    # Batch drug molecular graphs
    D_G = dgl.batch(D_graphs)
    
    # Stack sequence embeddings: [batch_size, seq_len, embed_dim]
    max_len = max([t.shape[0] for t in T0])
    T = torch.zeros(len(T0), max_len, T0[0].shape[1])
    for i, t in enumerate(T0):
        T[i, :t.shape[0], :] = t    
    # Batch target contact graphs
    T_G = dgl.batch(T_graphs)
    
    # Labels
    Y = torch.tensor(Y0, dtype=torch.long)
    
    return D_G, T, T_G, Y
