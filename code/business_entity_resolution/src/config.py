from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("BER_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
DATASET_DIR = Path(os.getenv("BER_DATASET_DIR", PROJECT_ROOT / "dataset"))
TRAIN_DIR = Path(os.getenv("BER_TRAIN_DIR", DATASET_DIR / "train"))
TEST_DIR = Path(os.getenv("BER_TEST_DIR", DATASET_DIR / "test"))
OUTPUT_DIR = Path(os.getenv("BER_OUTPUT_DIR", PROJECT_ROOT / "output"))
ARTIFACT_DIR = Path(os.getenv("BER_ARTIFACT_DIR", PROJECT_ROOT / "models" / "artifacts"))
TRAIN_SOURCE_FILES = {"source1": TRAIN_DIR / "train_source1.tsv", "source2": TRAIN_DIR / "train_source2.tsv", "source3": TRAIN_DIR / "train_source3.tsv"}
TEST_SOURCE_FILES = {"source1": TEST_DIR / "test_source1.tsv", "source2": TEST_DIR / "test_source2.tsv", "source3": TEST_DIR / "test_source3.tsv"}
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
EXPECTED_ID_PREFIX = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}
SEED = int(os.getenv("BER_SEED", "42"))
TSV_CHUNKSIZE = int(os.getenv("BER_TSV_CHUNKSIZE", "50000"))

SEMANTIC_FEATURES_ENABLED = os.getenv("BER_SEMANTIC_FEATURES_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
SEMANTIC_MODEL = os.getenv("BER_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHOICE_C_ENABLED = os.getenv("BER_CHOICE_C_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
CROSS_ENCODER_MODEL = os.getenv("BER_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("BER_CROSS_ENCODER_BATCH_SIZE", "32"))
CROSS_ENCODER_DEVICE = os.getenv("BER_CROSS_ENCODER_DEVICE", "auto")
AMBIGUITY_MARGIN = float(os.getenv("BER_AMBIGUITY_MARGIN", "0.08"))
NEAR_THRESHOLD_WINDOW = float(os.getenv("BER_NEAR_THRESHOLD_WINDOW", "0.08"))
MAX_RERANK_PAIRS = int(os.getenv("BER_MAX_RERANK_PAIRS", "200000"))
CROSS_ENCODER_WEIGHT = float(os.getenv("BER_CROSS_ENCODER_WEIGHT", "0.35"))
DEFAULT_THRESHOLD = float(os.getenv("BER_MATCH_THRESHOLD", "0.70"))


def ensure_directories() -> None:
    for path in (OUTPUT_DIR, ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)
