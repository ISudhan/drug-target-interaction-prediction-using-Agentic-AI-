import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import GNNBlocks, WGCN, MultiscaleCNN, GNNBlockDTI
from src.data import DTIDataset, collate_fn
from src.training.trainer import test
from configs.biosnap import BIOSNAPConfig


def main():
    parser = argparse.ArgumentParser(description="Test GNNBlockDTI model")
    parser.add_argument("--data_dir", type=str, default="official/dataset/BIOSNAP/random_CV5",
                        help="Path to dataset splits")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    config = BIOSNAPConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data features...")
    try:
        drug_graphs = torch.load(os.path.join("official/dataset/BIOSNAP", "XD.pt"), weights_only=False)
        target_embeddings = torch.load(os.path.join("official/dataset/BIOSNAP", "XT.pt"), weights_only=False)
        target_graphs = torch.load(os.path.join("official/dataset/BIOSNAP", "XT_G.pt"), weights_only=False)
    except FileNotFoundError as e:
        print(f"Error loading features: {e}. Please run preprocessing first.")
        sys.exit(1)

    print("Loading test split...")
    fold_dir = os.path.join(args.data_dir, "fold_1")
    test_data = torch.load(os.path.join(fold_dir, "test.pt"), weights_only=False)
    
    test_dataset = DTIDataset(test_data, drug_graphs, target_embeddings, target_graphs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             collate_fn=collate_fn, drop_last=True)
    
    # Initialize model
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
    
    print(f"Loading checkpoint from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    
    loss_fn = nn.CrossEntropyLoss()
    
    print("Running evaluation...")
    test_loss, test_metric = test(
        model=model, 
        data_iter=test_loader, 
        loss_fn=loss_fn, 
        device=device
    )
    
    print("\nEvaluation completed!")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics - AUROC: {test_metric[0]:.4f}, AUPR: {test_metric[1]:.4f}, "
          f"ACC: {test_metric[2]:.4f}, Precision: {test_metric[3]:.4f}, "
          f"Recall: {test_metric[4]:.4f}, F1: {test_metric[5]:.4f}")

if __name__ == "__main__":
    main()
