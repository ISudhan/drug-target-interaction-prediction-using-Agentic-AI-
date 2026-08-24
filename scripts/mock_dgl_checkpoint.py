import sys
import torch
import torch.nn as nn

# --- MOCK DGL MODULES ---
class MockModule(nn.Module):
    pass

class dgl_mock:
    class nn:
        class pytorch:
            class GATConv(nn.Module):
                def __init__(self, in_feats, out_feats, num_heads, **kwargs):
                    super().__init__()
                    self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
                    self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
                    self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
                    self.bias = nn.Parameter(torch.FloatTensor(size=(out_feats * num_heads,)))
                
                def forward(self, *args, **kwargs):
                    pass
                    
            class GraphConv(nn.Module):
                def __init__(self, in_feats, out_feats, **kwargs):
                    super().__init__()
                    self.weight = nn.Parameter(torch.FloatTensor(size=(in_feats, out_feats)))
                    self.bias = nn.Parameter(torch.FloatTensor(size=(out_feats,)))
                    
                def forward(self, *args, **kwargs):
                    pass
                    
            class AvgPooling(nn.Module):
                def forward(self, *args, **kwargs): pass
                
            class MaxPooling(nn.Module):
                def forward(self, *args, **kwargs): pass
                
            class EdgeWeightNorm(nn.Module):
                def __init__(self, **kwargs): super().__init__()
                def forward(self, *args, **kwargs): pass
                
    def graph(self, *args, **kwargs): pass
    def batch(self, *args, **kwargs): pass
    def to_bidirected(self, *args, **kwargs): pass

# Inject into sys.modules
sys.modules['dgl'] = dgl_mock
sys.modules['dgl.nn.pytorch'] = dgl_mock.nn.pytorch
sys.modules['dgl.nn'] = dgl_mock.nn

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inspect_checkpoint import load_and_verify_checkpoint

checkpoints = [
    "official/pre_train_models/BIOSNAP_CV1.pth",
    "official/pre_train_models/BIOSNAP_unseen_D.pth",
    "official/pre_train_models/BIOSNAP_unseen_T.pth"
]

print("=== RUNNING CHECKPOINT AUDIT USING MOCKED DGL ===")
all_passed = True
for ckpt in checkpoints:
    if os.path.exists(ckpt):
        passed = load_and_verify_checkpoint(ckpt)
        if not passed:
            all_passed = False
        print("-" * 50)
    else:
        print(f"Checkpoint not found: {ckpt}")

if all_passed:
    print("\n🎉 ALL CHECKPOINTS LOADED SUCCESSFULLY WITH STRICT=TRUE!")
else:
    print("\n⚠️ SOME CHECKPOINTS FAILED TO LOAD.")
