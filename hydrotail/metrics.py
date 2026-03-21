from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    metrics["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return metrics


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def quantile_metrics(
    y_true: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
    quantiles: Iterable[float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    quantiles = list(quantiles)
    for quantile in quantiles:
        metrics[f"pinball_q{quantile}"] = pinball_loss(y_true, quantile_predictions[quantile], quantile)

    lower = min(quantiles)
    upper = max(quantiles)
    lower_pred = quantile_predictions[lower]
    upper_pred = quantile_predictions[upper]
    metrics["interval_coverage"] = float(np.mean((y_true >= lower_pred) & (y_true <= upper_pred)))
    metrics["interval_width"] = float(np.mean(upper_pred - lower_pred))
    return metrics


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_hat = (y_prob >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics


def evaluate_tail_subset(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    mask = y_true >= threshold
    if not np.any(mask):
        return {"tail_count": 0}
    result = regression_metrics(y_true[mask], y_pred[mask])
    result["tail_count"] = int(mask.sum())
    return result
