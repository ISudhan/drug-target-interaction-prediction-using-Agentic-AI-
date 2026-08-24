import os
import sys
import torch
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable
collections.MutableMapping = collections.abc.MutableMapping

import sys
class MockModule:
    def __getattr__(self, name): return MockModule
    def __call__(self, *args, **kwargs): return MockModule
sys.modules['esm'] = MockModule()
sys.modules['transformers'] = MockModule()

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import GNNBlocks, WGCN, MultiscaleCNN, GNNBlockDTI
from src.data import DTIDataset, collate_fn
from torch.utils.data import DataLoader
from configs.biosnap import BIOSNAPConfig

def main():
    config = BIOSNAPConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("=== INITIALIZING REAL MODEL ===")
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
    
    checkpoints = [
        "official/pre_train_models/BIOSNAP_CV1.pth",
        "official/pre_train_models/BIOSNAP_unseen_D.pth",
        "official/pre_train_models/BIOSNAP_unseen_T.pth"
    ]
    
    all_passed = True
    print("\n=== TESTING STRICT=TRUE LOADING ===")
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            print(f"Loading {ckpt}...")
            state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"✅ SUCCESS: {os.path.basename(ckpt)} loaded with strict=True")
            except RuntimeError as e:
                print(f"❌ FAILED: strict=True loading failed for {os.path.basename(ckpt)}")
                print(e)
                all_passed = False
        else:
            print(f"Checkpoint not found: {ckpt}")
            all_passed = False
            
    if not all_passed:
        print("Stopping due to checkpoint failure.")
        sys.exit(1)
        
    print("\n=== RUNNING REAL FORWARD PASS ===")
    import pickle
    from src.data.preprocessing import smi_2_graph
    
    print("Loading one real drug from official dataset...")
    with open('official/dataset/BIOSNAP/drug_smi_raw.pkl', 'rb') as f:
        drug_smi = pickle.load(f)
    
    # Take the first drug and generate a REAL DGL graph using RDKit
    sample_drug_id = list(drug_smi.keys())[0]
    sample_smi = drug_smi[sample_drug_id]
    print(f"Sample drug SMILES: {sample_smi}")
    real_dgl_graph = smi_2_graph(sample_smi)
    
    # Generate dummy protein embeddings/graphs of correct shape
    # ProtBERT outputs 30-dim logits. Avg sequence length ~500.
    seq_len = 500
    dummy_target_embedding = torch.randn(seq_len, 30)
    # ESM-1b contact map graph (using complete graph for simplicity)
    src = torch.arange(seq_len).repeat_interleave(seq_len)
    dst = torch.arange(seq_len).repeat(seq_len)
    dummy_target_graph = dgl_mock.graph((src, dst)) if 'dgl_mock' in globals() else __import__('dgl').graph((src, dst))
    dummy_target_graph.edata['weight'] = torch.ones(seq_len * seq_len)
    dummy_target_graph.ndata['feat'] = dummy_target_embedding.clone()
    
    print("Executing forward pass on batch...")
    model.eval()
    model.to(device)
    
    # Batch them (batch size 1)
    D = __import__('dgl').batch([real_dgl_graph]).to(device)
    T = dummy_target_embedding.unsqueeze(0).to(device)
    T1 = __import__('dgl').batch([dummy_target_graph]).to(device)
    
    with torch.no_grad():
        output = model((D, T, T1))
        
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [1, 2]")
    
    if output.shape == (1, 2):
        print("✅ SUCCESS: Forward pass executed correctly with expected shapes.")
    else:
        print("❌ FAILED: Output shape mismatch.")
        sys.exit(1)
        
    print("\n🎉 ALL TESTS PASSED.")

if __name__ == "__main__":
    main()
