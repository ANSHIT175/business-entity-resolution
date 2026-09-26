from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureBuilder:
    def __init__(self) -> None:
        self.name_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.address_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.fitted = False

    def fit(self, frames: list) -> "TfidfFeatureBuilder":
        names, addresses = [], []
        for frame in frames:
            names.extend(frame.get("name_norm", frame.get("business_name", [])).tolist())
            addresses.extend(frame.get("address_norm", frame.get("business_address", [])).tolist())
        if not any(str(value).strip() for value in names): names = ["empty name"]
        if not any(str(value).strip() for value in addresses): addresses = ["empty address"]
        self.name_vectorizer.fit(names)
        self.address_vectorizer.fit(addresses)
        self.fitted = True
        return self

    @staticmethod
    def _cosines(left, right) -> np.ndarray:
        # TfidfVectorizer rows are L2-normalized by default; keep the operation sparse.
        return np.asarray(left.multiply(right).sum(axis=1)).ravel().astype(float)

    def transform_pairs(self, pairs):
        if not self.fitted:
            raise RuntimeError("TfidfFeatureBuilder.fit must be called first")
        names_a = pairs["s1_name_norm"].tolist(); names_b = pairs["target_name_norm"].tolist()
        addrs_a = pairs["s1_address_norm"].tolist(); addrs_b = pairs["target_address_norm"].tolist()
        na, nb = self.name_vectorizer.transform(names_a), self.name_vectorizer.transform(names_b)
        aa, ab = self.address_vectorizer.transform(addrs_a), self.address_vectorizer.transform(addrs_b)
        return {"name_tfidf": self._cosines(na, nb), "address_tfidf": self._cosines(aa, ab)}
