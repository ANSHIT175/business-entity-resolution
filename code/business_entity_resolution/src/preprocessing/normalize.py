from __future__ import annotations

import re
import unicodedata

import pandas as pd

_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(value: object, *, keep_numbers: bool = True) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold().replace("&", " and ")
    text = text.replace("–", "-").replace("—", "-")
    text = _NON_ALNUM_RE.sub(" ", text)
    if not keep_numbers:
        text = re.sub(r"\d+", " ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokens(value: object, *, keep_numbers: bool = True) -> list[str]:
    normalized = normalize_text(value, keep_numbers=keep_numbers)
    return normalized.split() if normalized else []


def numeric_tokens(value: object) -> set[str]:
    text = "" if value is None else str(value)
    return set(re.findall(r"\d+[a-zA-Z]?", text.casefold()))


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from .normalize_addresses import normalize_address
    from .normalize_names import normalize_name
    out = frame.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_text)
    return out
