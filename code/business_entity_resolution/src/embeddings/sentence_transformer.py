from __future__ import annotations

import numpy as np

class SentenceTransformerFeatures:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to enable semantic features") from exc
        kwargs = {} if device == "auto" else {"device": device}
        self.model = SentenceTransformer(model_name, **kwargs)
        self.cache: dict[str, np.ndarray] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            embeddings = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            self.cache.update(dict(zip(missing, np.asarray(embeddings))))
        return np.asarray([self.cache[text] for text in texts])

    def pair_similarities(self, pairs) -> dict[str, np.ndarray]:
        def cosine(left, right):
            return np.sum(left * right, axis=1)
        name_a = self._encode(pairs["s1_name_norm"].tolist()); name_b = self._encode(pairs["target_name_norm"].tolist())
        addr_a = self._encode(pairs["s1_address_norm"].tolist()); addr_b = self._encode(pairs["target_address_norm"].tolist())
        return {"name_semantic": cosine(name_a, name_b), "address_semantic": cosine(addr_a, addr_b)}
