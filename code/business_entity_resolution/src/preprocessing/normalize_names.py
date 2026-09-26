from __future__ import annotations

import re

from .normalize import normalize_text, tokens

_SUFFIXES = {"incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd"}
_SUFFIX_RE = re.compile(r"\b(" + "|".join(map(re.escape, _SUFFIXES)) + r")\b")


def normalize_name(value: object) -> str:
    text = normalize_text(value)
    return _SUFFIX_RE.sub(lambda m: _SUFFIXES[m.group(1)], text)


def name_tokens(value: object) -> list[str]:
    return tokens(normalize_name(value))
