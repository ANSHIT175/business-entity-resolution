from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ..blocking.candidate_generator import generate_candidates
from ..config import ARTIFACT_DIR, SEMANTIC_FEATURES_ENABLED, SEMANTIC_MODEL, SEED, TRAIN_GROUND_TRUTH, TRAIN_SOURCE_FILES
from ..data.loader import ground_truth_pairs, load_ground_truth, load_sources
from ..features.feature_builder import build_features, make_pair_frame
from ..features.tfidf_features import TfidfFeatureBuilder
from .lightgbm_model import PairModel


def _augment_hard_negatives(candidates: pd.DataFrame, source1: pd.DataFrame, targets: pd.DataFrame, truth: set[tuple[str, str]], limit_per_source1: int = 5) -> pd.DataFrame:
    existing = set(zip(candidates["source1_entity_id"], candidates["matched_entity_id"]))
    rows = []
    target_countries = targets["country"].astype(str).str.casefold().tolist()
    for s1_row, s1 in source1.reset_index(drop=True).iterrows():
        same_country = [idx for idx, country in enumerate(target_countries) if country and country == str(s1["country"]).casefold()]
        pool = same_country or list(range(len(targets)))
        added = 0
        for target_row in pool:
            pair = (str(s1["entity_id"]), str(targets.iloc[target_row]["entity_id"]))
            if pair in existing or pair in truth:
                continue
            target_source = "source3" if pair[1].startswith("S3-") else "source2"
            rows.append({"source1_entity_id": pair[0], "matched_entity_id": pair[1], "matched_source": target_source, "s1_row": s1_row, "target_row": target_row})
            existing.add(pair)
            added += 1
            if added >= limit_per_source1:
                break
    if not rows:
        return candidates
    return pd.concat([candidates, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["source1_entity_id", "matched_entity_id"])


def semantic_provider():
    if not SEMANTIC_FEATURES_ENABLED:
        return None
    try:
        from ..embeddings.sentence_transformer import SentenceTransformerFeatures
        return SentenceTransformerFeatures(SEMANTIC_MODEL)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Semantic features disabled: {exc}")
        return None


def train_components(*, save: bool = True) -> tuple[PairModel, TfidfFeatureBuilder, object]:
    sources = load_sources(TRAIN_SOURCE_FILES)
    truth = ground_truth_pairs(load_ground_truth(TRAIN_GROUND_TRUTH))
    model, tfidf, semantic, _ = fit_components(sources, truth)
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model.save(ARTIFACT_DIR / "lightgbm_pair_model.joblib")
        joblib.dump(tfidf, ARTIFACT_DIR / "tfidf_builder.joblib")
        (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps({"semantic_features": semantic is not None, "semantic_model": SEMANTIC_MODEL if semantic is not None else None, "feature_columns": model.feature_columns}, indent=2), encoding="utf-8")
    return model, tfidf, semantic


def fit_components(sources: dict[str, pd.DataFrame], truth: set[tuple[str, str]]) -> tuple[PairModel, TfidfFeatureBuilder, object, pd.DataFrame]:
    candidates = generate_candidates(sources["source1"], sources["source2"], sources["source3"])
    targets = pd.concat([sources["source2"], sources["source3"]], ignore_index=True)
    candidates = _augment_hard_negatives(candidates, sources["source1"], targets, truth)
    if candidates.empty:
        raise ValueError("Blocking generated no training candidates and no fallback negatives were available")
    pairs = make_pair_frame(sources["source1"], targets, candidates)
    tfidf = TfidfFeatureBuilder().fit([sources["source1"], targets])
    semantic = semantic_provider()
    features, columns = build_features(pairs, tfidf, semantic.pair_similarities(pairs) if semantic is not None else None)
    labels = [int((a, b) in truth) for a, b in zip(pairs["source1_entity_id"], pairs["matched_entity_id"])]
    model = PairModel(SEED).fit(features, labels, columns)
    return model, tfidf, semantic, candidates


def train_model() -> Path:
    train_components(save=True)
    return ARTIFACT_DIR / "lightgbm_pair_model.joblib"


if __name__ == "__main__":
    print(train_model())
