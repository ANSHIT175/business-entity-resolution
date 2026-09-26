from __future__ import annotations

from collections import defaultdict


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    denom = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denom if denom else 0.0


def evaluate_pairs(predicted: set[tuple[str, str]], truth: set[tuple[str, str]]) -> dict[str, float]:
    tp = len(predicted & truth); fp = len(predicted - truth); fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f0.5": fbeta(precision, recall)}


def predictions_to_pairs(predictions) -> set[tuple[str, str]]:
    result = set()
    for row in predictions.itertuples(index=False):
        for target in str(row.matched_entity_ids).split(","):
            target = target.strip()
            if target:
                result.add((str(row.source1_entity_id), target))
    return result
