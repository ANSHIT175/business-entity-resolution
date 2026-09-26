from __future__ import annotations

import pandas as pd

from ..config import (CHOICE_C_ENABLED, CROSS_ENCODER_BATCH_SIZE, CROSS_ENCODER_DEVICE, CROSS_ENCODER_MODEL,
                      CROSS_ENCODER_WEIGHT, DEFAULT_THRESHOLD, MAX_RERANK_PAIRS, AMBIGUITY_MARGIN, NEAR_THRESHOLD_WINDOW)
from ..features.feature_builder import build_features, make_pair_frame
from ..reranking.selector import select_ambiguous_pairs
from ..reranking.cross_encoder import SelectiveCrossEncoder


def score_candidates(source1, targets, candidates, pair_model, tfidf, semantic=None, *, choice_c_enabled=CHOICE_C_ENABLED, threshold=DEFAULT_THRESHOLD):
    pairs = make_pair_frame(source1, targets, candidates)
    if pairs.empty:
        empty = pairs.reindex(columns=["source1_entity_id", "matched_entity_id", "matched_source", "s1_row", "target_row", "lgbm_score", "final_score"])
        return empty, {"candidate_count": 0, "reranked_count": 0, "choice_c_fallback": False}
    features, columns = build_features(pairs, tfidf, semantic)
    scores = pair_model.predict_proba(features)
    scored = pairs[["source1_entity_id", "matched_entity_id", "matched_source", "s1_row", "target_row", "s1_name_norm", "s1_address_norm", "s1_country_norm", "target_name_norm", "target_address_norm", "target_country_norm"]].copy()
    scored["lgbm_score"] = scores; scored["final_score"] = scores
    for column in ("name_fuzzy", "address_fuzzy"):
        scored[column] = features[column].to_numpy()
    reranked_count = 0
    reranking_error = None
    if choice_c_enabled and not scored.empty:
        selected = select_ambiguous_pairs(scored, threshold, AMBIGUITY_MARGIN, NEAR_THRESHOLD_WINDOW, MAX_RERANK_PAIRS)
        if len(selected):
            try:
                encoder = SelectiveCrossEncoder(CROSS_ENCODER_MODEL, CROSS_ENCODER_BATCH_SIZE, CROSS_ENCODER_DEVICE)
                ce_scores = encoder.predict_pairs(scored.loc[selected])
                scored.loc[selected, "cross_encoder_score"] = ce_scores
                scored["cross_encoder_score"] = scored["cross_encoder_score"].fillna(scored["lgbm_score"])
                scored.loc[selected, "final_score"] = ((1 - CROSS_ENCODER_WEIGHT) * scored.loc[selected, "lgbm_score"] + CROSS_ENCODER_WEIGHT * scored.loc[selected, "cross_encoder_score"])
                reranked_count = len(selected)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                reranking_error = f"{type(exc).__name__}: {exc}"
    stats = {"candidate_count": len(scored), "reranked_count": reranked_count, "choice_c_fallback": bool(choice_c_enabled and reranking_error)}
    if reranking_error:
        stats["choice_c_error"] = reranking_error
    return scored, stats


def aggregate_predictions(scored: pd.DataFrame, source1_ids, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    accepted = scored[scored["final_score"] >= threshold]
    grouped = accepted.groupby("source1_entity_id")["matched_entity_id"].apply(lambda values: ",".join(dict.fromkeys(map(str, values)))) if not accepted.empty else pd.Series(dtype=str)
    rows = [{"source1_entity_id": str(s1), "matched_entity_ids": str(grouped.get(s1, ""))} for s1 in source1_ids]
    return pd.DataFrame(rows)
