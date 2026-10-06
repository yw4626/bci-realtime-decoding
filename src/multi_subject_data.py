"""通道对齐：解码或批处理时统一到固定通道宽度（右侧零填充或截断）。"""

import numpy as np


def pad_or_trunc_channels(x: np.ndarray, n_target: int) -> np.ndarray:
    """Right-pad with zeros if too narrow; truncate trailing channels if too wide."""
    c = x.shape[1]
    if c == n_target:
        return x
    if c < n_target:
        pad = n_target - c
        return np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
    return x[:, :n_target].copy()
