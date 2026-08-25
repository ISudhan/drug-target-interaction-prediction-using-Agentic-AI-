import os
import sys
import argparse
import pickle
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
    parser.add_argument("--task", type=str, default="BIOSNAP", help="Dataset/Task name")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold (0-4)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    args = parser.parse_args()

    config = BIOSNAPConfig()

    # Override config with args if provided
    if args.epochs is not None: config.epochs = args.epochs
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.lr is not None: config.lr = args.lr

    data_dir = args.data_dir if args.data_dir else os.path.join("official/dataset", args.task)
    device_str = args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    # Reproducibility matching official code
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

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

    print(f"Loading CV fold {args.fold} splits...")
    cv_path = os.path.join(data_dir, "random_CV5", f"data_CV{args.fold}.pkl")
    try:
        with open(cv_path, 'rb') as f:
            data = pickle.load(f)
        train_data, valid_data, test_data = data
    except FileNotFoundError as e:
        print(f"Error loading fold data: {e}")
        sys.exit(1)

    print(f"Data train: {len(train_data)}")
    print(f"Data valid: {len(valid_data)}")
    print(f"Data test:  {len(test_data)}")

    train_dataset = DTIDataset(train_data, drug_graph, target_embedding, target_graph)
    valid_dataset = DTIDataset(valid_data, drug_graph, target_embedding, target_graph)
    test_dataset = DTIDataset(test_data, drug_graph, target_embedding, target_graph)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    save_name = f"GNNBlockDTI_{args.task}_CV{args.fold}"

    print(f"Starting training on {device}...")
    train_losses, valid_losses, valid_metrics, test_best = train(
        model=model,
        epochs=config.epochs,
        train_iter=train_loader,
        valid_iter=valid_loader,
        test_iter=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        compare="max",
        device=device,
        name=save_name,
        save_dir=args.save_dir
    )

    print("\nTraining completed!")
    if test_best:
        print(f"Best Test Metrics - AUROC: {test_best[0]:.5f}, AUPR: {test_best[1]:.5f}, "
              f"ACC: {test_best[2]:.5f}, Precision: {test_best[3]:.5f}, "
              f"Recall: {test_best[4]:.5f}, F1: {test_best[5]:.5f}")

if __name__ == "__main__":
    main()
