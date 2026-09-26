from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd

from .config import (ARTIFACT_DIR, CHOICE_C_ENABLED, DEFAULT_THRESHOLD, OUTPUT_DIR, TEST_SOURCE_FILES,
                     TRAIN_SOURCE_FILES, ensure_directories)
from .blocking.candidate_generator import generate_candidates
from .data.loader import load_sources
from .features.feature_builder import make_pair_frame
from .models.lightgbm_model import PairModel
from .models.predict import aggregate_predictions, score_candidates
from .models.train import semantic_provider, train_components
from .evaluation.runner import run_source1_validation


def _load_or_train() -> tuple[PairModel, object, object]:
    model_path = ARTIFACT_DIR / "lightgbm_pair_model.joblib"
    tfidf_path = ARTIFACT_DIR / "tfidf_builder.joblib"
    if model_path.exists() and tfidf_path.exists():
        semantic = semantic_provider()
        metadata_path = ARTIFACT_DIR / "model_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("semantic_features", False):
                semantic = None
            elif semantic is None:
                print("Saved model used semantic features, but they are unavailable; retraining a Choice B-compatible model")
                return train_components(save=True)
        return PairModel.load(model_path), joblib.load(tfidf_path), semantic
    return train_components(save=True)


def run(*, test: bool = True, choice_c: bool = CHOICE_C_ENABLED, threshold: float = DEFAULT_THRESHOLD, tune_threshold: bool = False) -> dict:
    ensure_directories()
    if tune_threshold:
        validation = run_source1_validation(choice_c=choice_c)
        threshold = validation["threshold"]
        print({"validation_threshold": threshold, "validation_metrics": validation["metrics"], "validation_blocking_recall": validation["metrics"]["blocking_recall"]})
    model, tfidf, semantic = _load_or_train()
    sources = load_sources(TEST_SOURCE_FILES if test else TRAIN_SOURCE_FILES)
    candidates = generate_candidates(sources["source1"], sources["source2"], sources["source3"])
    targets = pd.concat([sources["source2"], sources["source3"]], ignore_index=True)
    candidates.to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)
    pairs = make_pair_frame(sources["source1"], targets, candidates)
    semantic_values = semantic.pair_similarities(pairs) if semantic is not None and not pairs.empty else None
    scored, stats = score_candidates(sources["source1"], targets, candidates, model, tfidf, semantic=semantic_values, choice_c_enabled=choice_c, threshold=threshold)
    predictions = aggregate_predictions(scored, sources["source1"]["entity_id"].tolist(), threshold)
    predictions.to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)
    result = {"choice_c_enabled": choice_c, **stats, "threshold": threshold, "output": str(OUTPUT_DIR)}
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Business entity resolution: Choice B baseline + optional Choice C")
    parser.add_argument("--train", action="store_true", help="train and score the training sources")
    parser.add_argument("--choice-c", action="store_true", help="enable selective Cross-Encoder reranking")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tune-threshold", action="store_true", help="tune threshold on a Source-1-level validation split before inference")
    args = parser.parse_args()
    run(test=not args.train, choice_c=args.choice_c, threshold=args.threshold, tune_threshold=args.tune_threshold)


if __name__ == "__main__":
    main()
