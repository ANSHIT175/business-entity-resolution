from __future__ import annotations

import numpy as np

from .metrics import evaluate_pairs


def tune_threshold(scored_pairs, truth: set[tuple[str, str]], thresholds=None) -> tuple[float, dict[str, float]]:
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 0.991, 0.01)
    best_threshold, best_metrics = 0.5, {"f0.5": -1.0}
    for threshold in thresholds:
        selected = scored_pairs[scored_pairs["final_score"] >= threshold]
        predicted = set(zip(selected["source1_entity_id"], selected["matched_entity_id"]))
        metrics = evaluate_pairs(predicted, truth)
        if metrics["f0.5"] > best_metrics["f0.5"] or (metrics["f0.5"] == best_metrics["f0.5"] and threshold > best_threshold):
            best_threshold, best_metrics = float(threshold), metrics
    return best_threshold, best_metrics
