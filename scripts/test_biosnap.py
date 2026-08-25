import os
import sys
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import GNNBlocks, WGCN, MultiscaleCNN, GNNBlockDTI
from src.data import DTIDataset, collate_fn
from src.training import test
from configs.biosnap import BIOSNAPConfig


def main():
    parser = argparse.ArgumentParser(description="Test GNNBlockDTI model on preprocessed BIOSNAP data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--task", type=str, default="BIOSNAP", help="Dataset/Task name")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold (0-4) to use for test split")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    args = parser.parse_args()
    
    config = BIOSNAPConfig()
    
    data_dir = args.data_dir if args.data_dir else os.path.join("official/dataset", args.task)
    device_str = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    
    print(f"Loading data features for {args.task}...")
    try:
        with open(os.path.join(data_dir, "drug_graph.pkl"), 'rb') as f:
            drug_graph = pickle.load(f)
        with open(os.path.join(data_dir, "target_embedding.pkl"), 'rb') as f:
            target_embedding = pickle.load(f)
        with open(os.path.join(data_dir, "target_graph.pkl"), 'rb') as f:
            target_graph = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading features: {e}. Please run preprocessing first.")
        sys.exit(1)

    print(f"Loading CV fold {args.fold} test split...")
    cv_path = os.path.join(data_dir, "random_CV5", f"data_CV{args.fold}.pkl")
    try:
        with open(cv_path, 'rb') as f:
            data = pickle.load(f)
        _, _, test_data = data  # We only need the test split
    except FileNotFoundError as e:
        print(f"Error loading fold data: {e}")
        sys.exit(1)
        
    print(f"Test samples: {len(test_data)}")
    
    test_dataset = DTIDataset(test_data, drug_graph, target_embedding, target_graph)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model components
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
    
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ SUCCESS: Model loaded with strict=True")
    except RuntimeError as e:
        print("❌ FAILED: strict=True loading failed.")
        print(f"Error: {e}")
        sys.exit(1)
        
    model.to(device)
    
    print(f"\nStarting evaluation on {device}...")
    loss_fn = nn.CrossEntropyLoss()

    test_loss, metrics = test(
        model,
        test_loader,
        loss_fn,
        device
    )
    
    print("\n--- Evaluation Results ---")
    print(f"AUROC:     {metrics[0]:.5f}")
    print(f"AUPR:      {metrics[1]:.5f}")
    print(f"Accuracy:  {metrics[2]:.5f}")
    print(f"Precision: {metrics[3]:.5f}")
    print(f"Recall:    {metrics[4]:.5f}")
    print(f"F1 Score:  {metrics[5]:.5f}")
    print(f"Test loss: {test_loss:.5f}")

if __name__ == "__main__":
    main()
