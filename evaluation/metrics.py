from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> Dict[str, float]:
    labels_np = np.asarray(list(labels), dtype=np.int64)
    probs_np = np.asarray(list(probabilities), dtype=np.float64)
    preds_np = (probs_np >= 0.5).astype(np.int64)

    metrics = {
        "Accuracy": float(accuracy_score(labels_np, preds_np)),
        "F1_score": float(f1_score(labels_np, preds_np, zero_division=0)),
        "Precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "Recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "AUC": 0.0,
    }

    if len(np.unique(labels_np)) == 2:
        metrics["AUC"] = float(roc_auc_score(labels_np, probs_np))
    return metrics

