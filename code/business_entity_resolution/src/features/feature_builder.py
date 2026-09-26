from __future__ import annotations

import pandas as pd

from ..preprocessing.normalize import normalize_frame
from .string_features import pair_string_features
from .tfidf_features import TfidfFeatureBuilder

FEATURE_COLUMNS = [
    "name_exact", "name_fuzzy", "name_jaccard", "name_token_recall", "name_length_ratio",
    "address_exact", "address_fuzzy", "address_jaccard", "address_token_recall",
    "shared_numeric", "numeric_jaccard", "country_exact", "name_token_count", "address_token_count",
    "name_tfidf", "address_tfidf", "name_semantic", "address_semantic",
]


def make_pair_frame(source1: pd.DataFrame, targets: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    left = normalize_frame(source1.reset_index(drop=True)).add_prefix("s1_")
    right = normalize_frame(targets.reset_index(drop=True)).add_prefix("target_")
    pairs = candidates.merge(left, left_on="s1_row", right_index=True, how="left").merge(right, left_on="target_row", right_index=True, how="left")
    return pairs


def build_features(pairs: pd.DataFrame, tfidf: TfidfFeatureBuilder, semantic: dict[str, object] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if pairs.empty:
        return pd.DataFrame(index=pairs.index), FEATURE_COLUMNS
    rows = [pair_string_features(row) for row in pairs.to_dict("records")]
    features = pd.DataFrame(rows, index=pairs.index)
    tfidf_values = tfidf.transform_pairs(pairs)
    for key, values in tfidf_values.items():
        features[key] = values
    features["name_semantic"] = 0.0
    features["address_semantic"] = 0.0
    if semantic:
        for key, values in semantic.items():
            if key in features:
                features[key] = values
    return features[FEATURE_COLUMNS].fillna(0.0), FEATURE_COLUMNS
