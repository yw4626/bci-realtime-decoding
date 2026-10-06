"""
流式友好的增量树回归：基于 Hoeffding 界（与 VFDT / Very Fast Decision Tree 同类思想）。

说明：经典 **VFDT** 论文针对**分类**；本任务为**连续标签**（五指位置），因此使用
``river.tree.HoeffdingTreeRegressor``（单样本 ``learn_one`` / ``predict_one``），
推理为 O(树深) 路径，通常比整片 Random Forest 更轻，且支持按样本继续更新。

依赖：``pip install -r requirements-streaming-tree.txt``
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from river import tree


def features_to_dict(row: np.ndarray) -> Dict[str, float]:
    r = np.asarray(row, dtype=np.float64).ravel()
    return {str(i): float(r[i]) for i in range(r.shape[0])}


class MultiFingerHoeffdingRegressor:
    """
    每指一棵 ``HoeffdingTreeRegressor``，与 ``MultiFingerModel`` 一样提供 ``fit`` / ``predict``，
    便于接入现有流水线与 joblib 持久化。
    """

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-6,
        leaf_prediction: str = "adaptive",
        max_size: int = 30,
        nominal_attributes: Optional[List] = None,
    ):
        self.grace_period = grace_period
        self.delta = delta
        self.leaf_prediction = leaf_prediction
        self.max_size = max_size
        self.nominal_attributes = nominal_attributes
        self.models: List[tree.HoeffdingTreeRegressor] = []

    def _make_tree(self) -> tree.HoeffdingTreeRegressor:
        kw = dict(
            grace_period=self.grace_period,
            delta=self.delta,
            leaf_prediction=self.leaf_prediction,
            max_size=self.max_size,
        )
        if self.nominal_attributes is not None:
            kw["nominal_attributes"] = self.nominal_attributes
        return tree.HoeffdingTreeRegressor(**kw)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiFingerHoeffdingRegressor":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_fingers = y.shape[1]
        self.models = [self._make_tree() for _ in range(n_fingers)]
        for fi in range(x.shape[0]):
            xd = features_to_dict(x[fi])
            for finger_idx, m in enumerate(self.models):
                m.learn_one(xd, float(y[fi, finger_idx]))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        out = np.zeros((n, len(self.models)), dtype=np.float64)
        for fi in range(n):
            xd = features_to_dict(x[fi])
            for finger_idx, m in enumerate(self.models):
                out[fi, finger_idx] = m.predict_one(xd)
        return out

    def learn_one_row(self, x_row: np.ndarray, y_row: np.ndarray) -> None:
        """在线单步更新（流式场景）：``x_row`` 一维特征向量，``y_row`` 形状 ``(5,)``。"""
        xd = features_to_dict(x_row)
        for finger_idx, m in enumerate(self.models):
            m.learn_one(xd, float(y_row[finger_idx]))
