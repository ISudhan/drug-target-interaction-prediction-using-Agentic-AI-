import os
import sys
import torch

# Add the project root to the path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import GNNBlockDTI, GNNBlocks, WGCN, MultiscaleCNN
from configs.biosnap import BIOSNAPConfig


def load_and_verify_checkpoint(checkpoint_path):
    """Loads an official checkpoint into the reconstructed model."""
    
    config = BIOSNAPConfig()
    
    # Reconstruct the model
    net_T = MultiscaleCNN(
        inputs=30, embed_dim=config.embed_dim, 
        nums_conv=[32, 64, 96], ksize=[5, 7, 13], 
        dropout=0.2
    )
    net_T1 = WGCN(input_size=30, hidden_size=config.hidden_size, dropout=0.2)
    net_D = GNNBlocks(
        input_size=64, hidden_size=96, 
        n_layer=config.n_layer, n_gnn=config.n_gnn, 
        dropout=config.dropout_d
    )
    
    model = GNNBlockDTI(
        net_D, net_T, net_T1, 
        Drug_len=config.drug_len, 
        Target_len=config.target_len, 
        Target_len1=config.target_len1,
        hid_size=config.hid_size, 
        dropout=config.dropout_DT
    )
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    
    print(f"Checkpoint keys: {len(ckpt_keys)}")
    print(f"Model keys: {len(model_keys)}")
    
    if missing:
        print(f"Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected}")
        
    shape_mismatches = 0
    for key in state_dict:
        if key in model.state_dict() and state_dict[key].shape != model.state_dict()[key].shape:
            print(f"Shape mismatch for {key}: ckpt {state_dict[key].shape} != model {model.state_dict()[key].shape}")
            shape_mismatches += 1
            
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ SUCCESS: strict=True loading passed!")
        print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
        return True
    except RuntimeError as e:
        print("❌ FAILED: strict=True loading failed.")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    checkpoints = [
        "models/BIOSNAP_CV1.pth",
        "models/BIOSNAP_D.pth",
        "models/BIOSNAP_T.pth"
    ]
    
    all_passed = True
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            passed = load_and_verify_checkpoint(ckpt)
            if not passed:
                all_passed = False
            print("-" * 50)
        else:
            print(f"Checkpoint not found: {ckpt}")
            all_passed = False
            
    if all_passed:
        print("\n🎉 ALL CHECKPOINTS LOADED SUCCESSFULLY WITH STRICT=TRUE!")
        sys.exit(0)
    else:
        print("\n⚠️ SOME CHECKPOINTS FAILED TO LOAD.")
        sys.exit(1)
