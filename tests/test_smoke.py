import unittest

import pandas as pd

from business_entity_resolution.src.blocking.candidate_generator import generate_candidates
from business_entity_resolution.src.features.feature_builder import build_features, make_pair_frame
from business_entity_resolution.src.features.tfidf_features import TfidfFeatureBuilder
from business_entity_resolution.src.models.predict import aggregate_predictions
from business_entity_resolution.src.reranking.selector import select_ambiguous_pairs


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.s1 = pd.DataFrame([{"entity_id": "S1-1", "business_name": "ABC Technologies Pvt Ltd", "business_address": "12 Main Street", "country": "France"}, {"entity_id": "S1-2", "business_name": "No Match Shop", "business_address": "Unknown", "country": "France"}])
        self.s2 = pd.DataFrame([{"entity_id": "S2-1", "business_name": "ABC Technology Private Limited", "business_address": "12 Main St", "country": "France"}])
        self.s3 = pd.DataFrame(columns=self.s2.columns)

    def test_blocking_and_features(self):
        candidates = generate_candidates(self.s1, self.s2, self.s3)
        targets = pd.concat([self.s2, self.s3], ignore_index=True)
        pairs = make_pair_frame(self.s1, targets, candidates)
        tfidf = TfidfFeatureBuilder().fit([self.s1, targets])
        features, columns = build_features(pairs, tfidf)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(features), 1)
        self.assertIn("name_tfidf", columns)
        self.assertEqual(pairs.iloc[0]["target_name_norm"], "abc technology private ltd")

    def test_all_source1_entities_are_aggregated(self):
        scored = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_id": ["S2-1"], "final_score": [0.9]})
        output = aggregate_predictions(scored, ["S1-1", "S1-2"], 0.7)
        self.assertEqual(output.to_dict("records"), [{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"}, {"source1_entity_id": "S1-2", "matched_entity_ids": ""}])

    def test_selector_only_selects_hard_pairs(self):
        scored = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-2"], "lgbm_score": [0.72, 0.70, 0.1], "name_fuzzy": [0.9, 0.8, 0.1], "address_fuzzy": [0.2, 0.8, 0.1]})
        selected = select_ambiguous_pairs(scored, 0.7, 0.05, 0.03, 100)
        self.assertEqual(set(selected), {0, 1})

    def test_empty_texts_do_not_break_tfidf(self):
        empty = pd.DataFrame([{"entity_id": "S1-1", "business_name": "", "business_address": "", "country": ""}])
        builder = TfidfFeatureBuilder().fit([empty])
        self.assertEqual(len(builder.transform_pairs(make_pair_frame(empty, empty, pd.DataFrame([{"source1_entity_id": "S1-1", "matched_entity_id": "S1-1", "matched_source": "source2", "s1_row": 0, "target_row": 0}])))["name_tfidf"]), 1)


if __name__ == "__main__":
    unittest.main()
