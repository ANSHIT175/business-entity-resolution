from __future__ import annotations

from difflib import SequenceMatcher

from ..preprocessing.normalize import numeric_tokens, tokens

try:
    from rapidfuzz.fuzz import ratio as _ratio
except ImportError:  # pragma: no cover
    _ratio = lambda a, b: 100 * SequenceMatcher(None, a, b).ratio()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def pair_string_features(row: dict) -> dict[str, float]:
    name_a, name_b = row["s1_name_norm"], row["target_name_norm"]
    addr_a, addr_b = row["s1_address_norm"], row["target_address_norm"]
    name_tokens_a, name_tokens_b = set(tokens(name_a)), set(tokens(name_b))
    addr_tokens_a, addr_tokens_b = set(tokens(addr_a)), set(tokens(addr_b))
    nums_a, nums_b = numeric_tokens(addr_a), numeric_tokens(addr_b)
    return {
        "name_exact": float(bool(name_a) and name_a == name_b),
        "name_fuzzy": _ratio(name_a, name_b) / 100.0,
        "name_jaccard": _jaccard(name_tokens_a, name_tokens_b),
        "name_token_recall": len(name_tokens_a & name_tokens_b) / len(name_tokens_a) if name_tokens_a else 0.0,
        "name_length_ratio": min(len(name_a), len(name_b)) / max(len(name_a), len(name_b), 1),
        "address_exact": float(bool(addr_a) and addr_a == addr_b),
        "address_fuzzy": _ratio(addr_a, addr_b) / 100.0,
        "address_jaccard": _jaccard(addr_tokens_a, addr_tokens_b),
        "address_token_recall": len(addr_tokens_a & addr_tokens_b) / len(addr_tokens_a) if addr_tokens_a else 0.0,
        "shared_numeric": float(bool(nums_a & nums_b)),
        "numeric_jaccard": _jaccard(nums_a, nums_b),
        "country_exact": float(row["s1_country_norm"] == row["target_country_norm"] and bool(row["s1_country_norm"])),
        "name_token_count": float(len(name_tokens_a)),
        "address_token_count": float(len(addr_tokens_a)),
    }
