from __future__ import annotations

from ..config import SOURCE_COLUMNS


def validate_source_frame(frame) -> None:
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if frame["entity_id"].astype(str).str.strip().eq("").any():
        raise ValueError("entity_id cannot be empty")
    if frame["entity_id"].duplicated().any():
        raise ValueError("entity_id must be unique within each source")
