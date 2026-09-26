from __future__ import annotations

import numpy as np

from .metrics import evaluate_pairs


def split_source1_entities(source1, validation_fraction: float = 0.2, seed: int = 42):
    """Split whole S1 entities, preventing candidate-pair leakage across folds."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    ids = np.asarray(source1["entity_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_validation = max(1, min(len(ids) - 1, int(round(len(ids) * validation_fraction)))) if len(ids) > 1 else 0
    validation_ids = set(ids[:n_validation])
    train_ids = set(ids[n_validation:])
    return train_ids, validation_ids


def filter_candidates_by_source1(candidates, source1_ids):
    return candidates[candidates["source1_entity_id"].astype(str).isin(set(map(str, source1_ids)))].copy()


def blocking_recall(candidates, truth: set[tuple[str, str]]) -> float:
    available = set(zip(candidates.get("source1_entity_id", []), candidates.get("matched_entity_id", [])))
    return len(available & truth) / len(truth) if truth else 1.0


def validate_scored_pairs(scored_pairs, truth: set[tuple[str, str]], threshold: float) -> dict[str, float]:
    selected = scored_pairs[scored_pairs["final_score"] >= threshold]
    predicted = set(zip(selected["source1_entity_id"], selected["matched_entity_id"]))
    return evaluate_pairs(predicted, truth)


def validation_summary(candidates, scored_pairs, truth: set[tuple[str, str]], threshold: float) -> dict[str, float]:
    metrics = validate_scored_pairs(scored_pairs, truth, threshold)
    metrics["blocking_recall"] = blocking_recall(candidates, truth)
    return metrics
