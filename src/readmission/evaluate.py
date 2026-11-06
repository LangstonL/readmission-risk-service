"""Binary classification metrics for the readmission baseline."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Report ranking, calibration, and prevalence metrics."""
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "prevalence": float(np.mean(y_true)),
        "n": len(y_true),
    }
