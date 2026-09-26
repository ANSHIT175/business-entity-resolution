from __future__ import annotations

import pandas as pd

from ..blocking.candidate_generator import generate_candidates
from ..config import TRAIN_GROUND_TRUTH, TRAIN_SOURCE_FILES
from ..data.loader import ground_truth_pairs, load_ground_truth, load_sources
from ..features.feature_builder import make_pair_frame
from ..models.predict import score_candidates
from ..models.train import fit_components
from .threshold_tuning import tune_threshold
from .validation import blocking_recall, split_source1_entities, validate_scored_pairs


def run_source1_validation(validation_fraction: float = 0.2, seed: int = 42, *, choice_c: bool = False) -> dict:
    sources = load_sources(TRAIN_SOURCE_FILES)
    truth = ground_truth_pairs(load_ground_truth(TRAIN_GROUND_TRUTH))
    train_ids, validation_ids = split_source1_entities(sources["source1"], validation_fraction, seed)
    train_s1 = sources["source1"][sources["source1"]["entity_id"].isin(train_ids)].reset_index(drop=True)
    validation_s1 = sources["source1"][sources["source1"]["entity_id"].isin(validation_ids)].reset_index(drop=True)
    train_sources = {**sources, "source1": train_s1}
    model, tfidf, semantic, _ = fit_components(train_sources, truth)
    candidates = generate_candidates(validation_s1, sources["source2"], sources["source3"])
    targets = pd.concat([sources["source2"], sources["source3"]], ignore_index=True)
    pairs = make_pair_frame(validation_s1, targets, candidates)
    semantic_values = semantic.pair_similarities(pairs) if semantic is not None and not pairs.empty else None
    validation_truth = {(left, right) for left, right in truth if left in validation_ids}
    scored, stats = score_candidates(validation_s1, targets, candidates, model, tfidf, semantic=semantic_values, choice_c_enabled=choice_c)
    threshold, tuned_metrics = tune_threshold(scored, validation_truth)
    metrics = validate_scored_pairs(scored, validation_truth, threshold)
    metrics["blocking_recall"] = blocking_recall(candidates, validation_truth)
    return {"threshold": threshold, "metrics": metrics, "candidate_count": len(candidates), "reranker": stats}
