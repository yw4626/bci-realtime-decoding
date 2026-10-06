from typing import List

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MultiFingerMLP:
    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.models: List[MLPRegressor] = []
        self.x_scaler: StandardScaler = StandardScaler()
        self.y_scalers: List[StandardScaler] = []

    def _make_model(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiFingerMLP":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xs = self.x_scaler.fit_transform(x)
        self.models = []
        self.y_scalers = []
        for finger_idx in range(y.shape[1]):
            ys = StandardScaler()
            yt = ys.fit_transform(y[:, [finger_idx]]).ravel()
            m = self._make_model()
            m.fit(xs, yt)
            self.models.append(m)
            self.y_scalers.append(ys)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xs = self.x_scaler.transform(x)
        preds = []
        for i, m in enumerate(self.models):
            pz = m.predict(xs).reshape(-1, 1)
            p = self.y_scalers[i].inverse_transform(pz).ravel()
            preds.append(p)
        return np.vstack(preds).T
