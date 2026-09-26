from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def validate(output_dir: Path, test_dir: Path) -> list[str]:
    errors = []
    results_path, candidates_path = output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv"
    if not results_path.exists(): errors.append(f"missing {results_path}")
    if not candidates_path.exists(): errors.append(f"missing {candidates_path}")
    if errors: return errors
    results = pd.read_csv(results_path, sep="\t", dtype=str, keep_default_na=False)
    candidates = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)
    if list(results.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errors.append("invalid matching_results columns")
        return errors
    required_candidate_columns = {"source1_entity_id", "matched_entity_id"}
    if not required_candidate_columns.issubset(candidates.columns):
        errors.append(f"candidate_pairs is missing columns: {sorted(required_candidate_columns - set(candidates.columns))}")
        return errors
    if results.source1_entity_id.duplicated().any(): errors.append("duplicate Source 1 rows")
    if candidates.duplicated(["source1_entity_id", "matched_entity_id"]).any(): errors.append("duplicate candidate pairs")
    valid_ids = set()
    expected_s1 = set()
    s1_path = test_dir / "test_source1.tsv"
    if s1_path.exists(): expected_s1.update(pd.read_csv(s1_path, sep="\t", dtype=str)["entity_id"].astype(str))
    if expected_s1 != set(results.source1_entity_id): errors.append("matching_results must contain every Source 1 entity exactly once")
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = test_dir / name
        if path.exists(): valid_ids.update(pd.read_csv(path, sep="\t", dtype=str)["entity_id"].astype(str))
    if valid_ids:
        for candidate in candidates.itertuples(index=False):
            if candidate.matched_entity_id not in valid_ids: errors.append(f"invalid candidate target id {candidate.matched_entity_id}")
            if candidate.source1_entity_id == candidate.matched_entity_id: errors.append(f"candidate self-match for {candidate.source1_entity_id}")
            if expected_s1 and candidate.source1_entity_id not in expected_s1: errors.append(f"invalid candidate Source 1 id {candidate.source1_entity_id}")
    candidate_set = set(zip(candidates.source1_entity_id, candidates.matched_entity_id))
    for row in results.itertuples(index=False):
        ids = [x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()]
        if len(ids) != len(set(ids)): errors.append(f"duplicate matches for {row.source1_entity_id}")
        for target in ids:
            if target == row.source1_entity_id: errors.append(f"self-match for {row.source1_entity_id}")
            if valid_ids and target not in valid_ids: errors.append(f"invalid target id {target}")
            if (row.source1_entity_id, target) not in candidate_set: errors.append(f"prediction not in candidates: {row.source1_entity_id},{target}")
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    args = parser.parse_args()
    errors = validate(args.output_dir, args.test_dir)
    if errors:
        print("INVALID")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("VALID")
