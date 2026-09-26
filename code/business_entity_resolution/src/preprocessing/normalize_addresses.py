from __future__ import annotations

import re

from .normalize import normalize_text, tokens

_ABBREVIATIONS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "apartment": "apt",
    "suite": "ste", "building": "bldg", "number": "no",
}
_ABBR_RE = re.compile(r"\b(" + "|".join(map(re.escape, _ABBREVIATIONS)) + r")\b")


def normalize_address(value: object) -> str:
    text = normalize_text(value, keep_numbers=True)
    return _ABBR_RE.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)


def address_tokens(value: object) -> list[str]:
    return tokens(normalize_address(value), keep_numbers=True)
