from typing import Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def cosine_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - (np.dot(a, b) / denom))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def eval_finger_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "cosine_distance": cosine_distance(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
    }

