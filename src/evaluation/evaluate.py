import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, f1_score


def estimate(y_true, Y_pred):
    """Calculate evaluation metrics for DTI prediction.
    
    Args:
        y_true: True labels (numpy array)
        Y_pred: Predicted logits [batch_size, 2] (numpy array or tensor)
        
    Returns:
        tuple: (AUROC, AUPR, Accuracy, Precision, Recall, F1)
    """
    if not isinstance(Y_pred, torch.Tensor):
        Y_pred = torch.tensor(Y_pred)
        
    Y_pred = F.softmax(Y_pred, 1) 
    Y_pred = Y_pred.cpu().numpy()
    
    Y_pred_label = np.argmax(Y_pred, axis=1)
    Y_pred_score = Y_pred[:, 1]
    
    try:
        auroc = round(roc_auc_score(y_true, Y_pred_score), ndigits=5)
    except ValueError:
        auroc = 0.0
        
    tpr, fpr, _ = precision_recall_curve(y_true, Y_pred_score)
    aupr = round(auc(fpr, tpr), ndigits=5)
    
    Precision = precision_score(y_true, Y_pred_label, zero_division=0)
    Recall = recall_score(y_true, Y_pred_label, zero_division=0)
    Accuracy = accuracy_score(y_true, Y_pred_label)
    F1 = f1_score(y_true, Y_pred_label, zero_division=0)

    return auroc, aupr, Accuracy, Precision, Recall, F1 
