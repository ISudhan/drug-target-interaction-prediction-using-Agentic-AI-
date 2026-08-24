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
from src.training import train
from configs.biosnap import BIOSNAPConfig


def main():
    parser = argparse.ArgumentParser(description="Train GNNBlockDTI model")
    parser.add_argument("--data_dir", type=str, default="official/dataset/BIOSNAP/random_CV5",
                        help="Path to dataset splits")
    parser.add_argument("--save_name", type=str, default="GNNBlockDTI_CV",
                        help="Name prefix for saving the model checkpoint")
    args = parser.parse_args()
    
    config = BIOSNAPConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset features (using precomputed features from the official implementation)
    # Note: In a full pipeline, these would be computed via src/data/preprocessing.py
    # Here we assume the features exist as in the official code.
    print("Loading data features...")
    try:
        drug_graphs = torch.load(os.path.join("official/dataset/BIOSNAP", "XD.pt"), weights_only=False)
        target_embeddings = torch.load(os.path.join("official/dataset/BIOSNAP", "XT.pt"), weights_only=False)
        target_graphs = torch.load(os.path.join("official/dataset/BIOSNAP", "XT_G.pt"), weights_only=False)
    except FileNotFoundError as e:
        print(f"Error loading features: {e}. Please run preprocessing first.")
        sys.exit(1)

    print("Loading splits...")
    fold_dir = os.path.join(args.data_dir, "fold_1")
    train_data = torch.load(os.path.join(fold_dir, "train.pt"), weights_only=False)
    valid_data = torch.load(os.path.join(fold_dir, "valid.pt"), weights_only=False)
    test_data = torch.load(os.path.join(fold_dir, "test.pt"), weights_only=False)
    
    train_dataset = DTIDataset(train_data, drug_graphs, target_embeddings, target_graphs)
    valid_dataset = DTIDataset(valid_data, drug_graphs, target_embeddings, target_graphs)
    test_dataset = DTIDataset(test_data, drug_graphs, target_embeddings, target_graphs)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              collate_fn=collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, 
                              collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             collate_fn=collate_fn, drop_last=True)
    
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
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    print("Starting training...")
    train_losses, valid_losses, valid_metrics, test_best = train(
        model=model, 
        epochs=config.epochs, 
        train_iter=train_loader, 
        valid_iter=valid_loader, 
        test_iter=test_loader, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        compare="max",  # Select based on max AUROC
        device=device, 
        name=args.save_name
    )
    
    print("\nTraining completed!")
    if test_best:
        print(f"Best Test Metrics - AUROC: {test_best[0]:.4f}, AUPR: {test_best[1]:.4f}, "
              f"ACC: {test_best[2]:.4f}, Precision: {test_best[3]:.4f}, "
              f"Recall: {test_best[4]:.4f}, F1: {test_best[5]:.4f}")

if __name__ == "__main__":
    main()
