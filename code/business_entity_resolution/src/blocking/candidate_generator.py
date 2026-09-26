from __future__ import annotations

from collections import defaultdict
import pandas as pd
from ..preprocessing.normalize import normalize_frame, tokens


def _index(rows: pd.DataFrame, key: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for idx, value in rows[key].items():
        if value:
            result[value].append(idx)
    return result


def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
    """Generate unioned exact/token/address blocks without a Cartesian product."""
    s1 = normalize_frame(source1.reset_index(drop=True))
    targets, target_offset = [], 0
    for source_name, frame in (("source2", source2), ("source3", source3)):
        current = normalize_frame(frame.reset_index(drop=True)).copy()
        current["matched_source"] = source_name
        current["target_row"] = range(target_offset, target_offset + len(current))
        target_offset += len(current)
        targets.append(current)
    target = pd.concat(targets, ignore_index=True) if targets else pd.DataFrame()
    columns = ["source1_entity_id", "matched_entity_id", "matched_source", "s1_row", "target_row"]
    if s1.empty or target.empty:
        return pd.DataFrame(columns=columns)
    blocks = {f"target_{key}": _index(target, key) for key in ("name_norm", "address_norm", "country_norm")}
    token_index: dict[str, list[int]] = defaultdict(list)
    addr_token_index: dict[str, list[int]] = defaultdict(list)
    for idx, row in target.iterrows():
        for token in set(tokens(row["name_norm"])):
            if len(token) >= 3: token_index[token].append(idx)
        for token in set(tokens(row["address_norm"])):
            if len(token) >= 3: addr_token_index[token].append(idx)
    records: set[tuple[int, int]] = set()
    for s1_idx, row in s1.iterrows():
        target_ids = set(blocks["target_name_norm"].get(row["name_norm"], []))
        target_ids.update(blocks["target_address_norm"].get(row["address_norm"], []))
        name_tokens = set(tokens(row["name_norm"]))
        address_tokens = set(tokens(row["address_norm"]))
        for token in name_tokens:
            if len(token) >= 3: target_ids.update(token_index.get(token, []))
        for token in address_tokens:
            if len(token) >= 3: target_ids.update(addr_token_index.get(token, []))
        # Country is only an enrichment block, never an exclusion filter.
        if row["country_norm"] and name_tokens:
            same_country = set(blocks["target_country_norm"].get(row["country_norm"], []))
            for token in name_tokens:
                if len(token) >= 3:
                    target_ids.update(same_country.intersection(token_index.get(token, [])))
        records.update((s1_idx, idx) for idx in target_ids)
    output = []
    for s1_idx, target_idx in sorted(records):
        s1_row, target_row = s1.iloc[s1_idx], target.iloc[target_idx]
        output.append({"source1_entity_id": s1_row.entity_id, "matched_entity_id": target_row.entity_id, "matched_source": target_row.matched_source, "s1_row": s1_idx, "target_row": target_idx})
    return pd.DataFrame(output, columns=columns).drop_duplicates(["source1_entity_id", "matched_entity_id"])
