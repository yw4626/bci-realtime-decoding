from typing import Tuple

import numpy as np
from scipy.io import loadmat


def load_bci_mat(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(path)
    train_x = data["train_data"].astype(np.float64)
    train_y = data["train_dg"].astype(np.float64)
    test_x = data["test_data"].astype(np.float64)
    return train_x, train_y, test_x


def temporal_train_val_split(
    x: np.ndarray, y: np.ndarray, val_fraction: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cut = int(len(x) * (1 - val_fraction))
    return x[:cut], y[:cut], x[cut:], y[cut:]
