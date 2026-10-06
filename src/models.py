from typing import Dict, List, Optional

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class MultiFingerModel:
    def __init__(self, model_type: str, random_state: int = 42, ridge_alpha: float = 1.0):
        self.model_type = model_type
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha
        self.models: List = []
        # simple：与 MultiFingerMLP 一致，全训练集上共用一个 StandardScaler(X)；各指仅对 y 做标准化 + Ridge。
        self._x_scaler: Optional[StandardScaler] = None

    def _make_model(self):
        if self.model_type == "simple":
            return TransformedTargetRegressor(
                regressor=Ridge(alpha=self.ridge_alpha),
                transformer=StandardScaler(),
            )
        if self.model_type == "complex":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiFingerModel":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.models = []
        if self.model_type == "simple":
            self._x_scaler = StandardScaler()
            x_in = self._x_scaler.fit_transform(x)
        else:
            self._x_scaler = None
            x_in = x
        for finger_idx in range(y.shape[1]):
            m = self._make_model()
            m.fit(x_in, y[:, finger_idx])
            self.models.append(m)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scaler = getattr(self, "_x_scaler", None)
        if self.model_type == "simple" and scaler is not None:
            x = scaler.transform(x)
        preds = [m.predict(x) for m in self.models]
        return np.vstack(preds).T


def blend_predictions(simple_pred: np.ndarray, complex_pred: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * complex_pred + (1 - alpha) * simple_pred


def evaluate_model_family(
    simple_model: MultiFingerModel,
    complex_model: MultiFingerModel,
    x_val: np.ndarray,
) -> Dict[str, np.ndarray]:
    simple_pred = simple_model.predict(x_val)
    complex_pred = complex_model.predict(x_val)
    blended_pred = blend_predictions(simple_pred, complex_pred, alpha=0.6)
    return {"simple": simple_pred, "complex": complex_pred, "blended": blended_pred}

