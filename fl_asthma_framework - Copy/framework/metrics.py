import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def safe_auc(y_true, probs):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, probs)

def evaluate_binary(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = safe_auc(y_true, probs)
    cm = confusion_matrix(y_true, preds)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "cm": cm.tolist(),
    }
