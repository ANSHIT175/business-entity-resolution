from __future__ import annotations

import pandas as pd


def select_ambiguous_pairs(scored_pairs: pd.DataFrame, threshold: float, margin: float, near_window: float, max_pairs: int) -> pd.Index:
    if scored_pairs.empty:
        return scored_pairs.index
    selected: set[int] = set()
    for _, group in scored_pairs.groupby("source1_entity_id", sort=False):
        ordered = group.sort_values("lgbm_score", ascending=False)
        scores = ordered["lgbm_score"].to_numpy()
        if len(scores) > 1 and scores[0] - scores[1] <= margin:
            selected.update(ordered.index.tolist())
        near = ordered[(ordered["lgbm_score"] >= threshold - near_window) & (ordered["lgbm_score"] <= threshold + near_window)]
        selected.update(near.index.tolist())
        # Conflicting evidence is a hard case even when the calibrated score is high.
        conflict = ordered[(ordered["name_fuzzy"] >= 0.85) & (ordered["address_fuzzy"] <= 0.35)]
        selected.update(conflict.index.tolist())
    if len(selected) > max_pairs:
        ranked = scored_pairs.loc[list(selected)].assign(_priority=lambda x: (x["lgbm_score"] - threshold).abs()).sort_values("_priority")
        selected = set(ranked.head(max_pairs).index)
    return pd.Index(sorted(selected))
