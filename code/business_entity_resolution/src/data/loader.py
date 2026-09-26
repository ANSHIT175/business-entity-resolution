from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from ..config import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, TSV_CHUNKSIZE
from .validation import validate_source_frame


def load_source(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing source file: {path}")
        return pd.DataFrame(columns=list(SOURCE_COLUMNS))
    chunks = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=TSV_CHUNKSIZE)
    frame = pd.concat(chunks, ignore_index=True)
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[:, list(SOURCE_COLUMNS)].fillna("")
    validate_source_frame(frame)
    return frame


def load_sources(paths: Dict[str, Path], *, required: bool = True) -> Dict[str, pd.DataFrame]:
    return {name: load_source(path, required=required) for name, path in paths.items()}


def load_ground_truth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ground truth: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(GROUND_TRUTH_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.loc[:, list(GROUND_TRUTH_COLUMNS)].fillna("")


def ground_truth_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        s1 = str(row.source1_entity_id).strip()
        for matched in str(row.matched_entity_ids).split(","):
            matched = matched.strip()
            if s1 and matched:
                pairs.add((s1, matched))
    return pairs
