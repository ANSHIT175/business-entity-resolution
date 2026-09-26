from __future__ import annotations

import joblib
import numpy as np

class PairModel:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.feature_columns: list[str] = []
        self.backend = "lightgbm"
        try:
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=seed, verbosity=-1)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.backend = "sklearn"
            self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)

    def fit(self, x, y, feature_columns: list[str]) -> "PairModel":
        self.feature_columns = list(feature_columns)
        labels = np.asarray(y, dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError("Training needs both positive and negative candidate pairs")
        self.model.fit(x[self.feature_columns], labels)
        return self

    def predict_proba(self, x) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x[self.feature_columns])[:, 1], dtype=float)

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "PairModel":
        return joblib.load(path)
