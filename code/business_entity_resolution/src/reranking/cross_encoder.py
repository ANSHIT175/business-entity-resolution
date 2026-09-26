from __future__ import annotations

import numpy as np

class SelectiveCrossEncoder:
    def __init__(self, model_name: str, batch_size: int = 32, device: str = "auto") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to enable Choice C") from exc
        kwargs = {"batch_size": batch_size}
        if device != "auto":
            kwargs["device"] = device
        self.model = CrossEncoder(model_name, **kwargs)
        self.batch_size = batch_size

    @staticmethod
    def _probability(scores) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if values.ndim == 2 and values.shape[1] == 2:
            shifted = values - values.max(axis=1, keepdims=True)
            exp_values = np.exp(np.clip(shifted, -30, 30))
            return (exp_values[:, 1] / exp_values.sum(axis=1)).astype(float)
        values = values.reshape(-1)
        # Cross-Encoder classifiers may return logits or probabilities.
        if np.all((values >= 0) & (values <= 1)):
            return values
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))

    def predict_pairs(self, pairs) -> np.ndarray:
        texts = [[f"business name: {a} address: {aa} country: {ac}", f"business name: {b} address: {ba} country: {bc}"] for a, aa, ac, b, ba, bc in zip(pairs["s1_name_norm"], pairs["s1_address_norm"], pairs["s1_country_norm"], pairs["target_name_norm"], pairs["target_address_norm"], pairs["target_country_norm"])]
        return self._probability(self.model.predict(texts, batch_size=self.batch_size, show_progress_bar=False))
