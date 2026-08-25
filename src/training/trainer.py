import os
import time
import torch
import numpy as np

from src.evaluation import estimate


def test(model, data_iter, loss_fn, device):
    """Evaluate model on a dataset.
    
    Args:
        model: GNNBlockDTI model
        data_iter: DataLoader for validation/test data
        loss_fn: Loss function (CrossEntropyLoss)
        device: Torch device
        
    Returns:
        tuple: (average_loss, metrics_tuple)
    """
    model.to(device)
    model.eval()
    
    test_L = []
    Y_preds = []
    Ys = []
    
    for _, (D, T, T1, Y) in enumerate(data_iter):
        D = D.to(device)
        T1 = T1.to(device)
        T = T.to(device)
        
        with torch.no_grad():
            Y_pred = model((D, T, T1))
            L = loss_fn(Y_pred.cpu().to(torch.float32), Y.to(torch.long))
            
            test_L.append(L.item())
            Y_preds.append(Y_pred.cpu())
            Ys.append(Y)
            
    test_loss = np.mean(test_L)
    
    Y_preds = torch.cat(Y_preds, dim=0)
    Ys = torch.cat(Ys, dim=0).numpy()
    
    metrics = estimate(Ys, Y_preds)

    return test_loss, metrics


def train(model, epochs, train_iter, valid_iter, test_iter, loss_fn, optimizer, compare, device, name, save_dir='models'):
    """Train the model.
    
    Args:
        model: GNNBlockDTI model
        epochs: Number of training epochs
        train_iter: Training DataLoader
        valid_iter: Validation DataLoader 
        test_iter: Test DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        compare: 'max' or 'min' for model selection based on primary metric
        device: Torch device
        name: Name for saving checkpoints
        save_dir: Directory to save model checkpoints
        
    Returns:
        tuple: (train_losses, valid_losses, valid_metrics, best_test_metrics)
    """
    model.to(device)
    print('Running on', device)
    
    train_L = []
    valid_L = []  # BUG FIX: Official code missed this initialization
    Metrics = []
    test_best = None
    best_epoch = 0 
    
    metrs_best = 10000 if compare == "min" else -10
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("--------------------------------------", name.replace('_', '——'))
    
    for epoch in range(epochs):
        train_L_1epoch = []
        start = time.time()
        
        model.train()
        for step, (D, T, T1, Y) in enumerate(train_iter):
            D = D.to(device)
            T1 = T1.to(device)
            T = T.to(device)
            
            optimizer.zero_grad()
            Y_pred = model((D, T, T1))
            L = loss_fn(Y_pred.to(torch.float32), Y.to(device, dtype=torch.long))
            
            L.backward()
            optimizer.step()
            
            train_L_1epoch.append(L.item())
            
        # Evaluation phase
        with torch.no_grad():
            train_loss_avg = np.mean(train_L_1epoch)
            train_L.append(train_loss_avg)
            
        valid_loss, valid_metric = test(model, valid_iter, loss_fn, device)
        valid_L.append(valid_loss)
        Metrics.append(valid_metric)
        
        # BUG FIX: Official code used undefined variable 'valid__L' here
        print(f'epoch: {epoch+1} train_loss: {train_L[-1]:.3f} valid_loss: {valid_L[-1]:.3f}')
        
        # Primary metric is AUROC (index 0)
        metrs = valid_metric[0]
        
        is_better = (metrs < metrs_best) if compare == "min" else (metrs > metrs_best)
        
        if is_better:
            metrs_best = metrs
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, f"{name}.pth"))
            print(f'The best model in epoch {epoch+1} has been saved!!! ')
            
            _, test_metric = test(model, test_iter, loss_fn, device)
            test_best = test_metric
            print("test metric (AUROC, AUPR, ACC, Precision, Recall, F1):", test_metric)

        time_epoch = time.time() - start
        
    torch.cuda.empty_cache()

    return train_L, valid_L, Metrics, test_best
