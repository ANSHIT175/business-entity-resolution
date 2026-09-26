# Business Entity Resolution

A from-scratch implementation of the competition pipeline. The repository intentionally excludes the large competition dataset.

## Pipeline

`TSV -> normalization -> multi-block candidates -> string/TF-IDF/semantic features -> LightGBM -> threshold -> S1 aggregation`

Choice B is the baseline. Sentence Transformer embeddings are optional and cached per unique text. Choice C is opt-in: after LightGBM scoring, only ambiguous/hard candidates are sent to a Cross-Encoder. It never runs over the Cartesian product and it never changes candidate generation.

## Layout

```text
business-entity-resolution/
  code/business_entity_resolution/src/
    config.py data/ preprocessing/ blocking/ features/ embeddings/
    models/ evaluation/ reranking/ pipeline.py
  dataset/                  # place competition files here; not included
  output/
  models/artifacts/
```

## Install and run

```bash
cd business-entity-resolution
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/code"
# Choice B baseline (Choice C disabled by default)
python -m business_entity_resolution.src.pipeline
# Explicit Choice C
python -m business_entity_resolution.src.pipeline --choice-c
# Tune F0.5 threshold on a Source-1-level validation split before inference
python -m business_entity_resolution.src.pipeline --tune-threshold
```

Training files are read from `dataset/train` and test files from `dataset/test`. Override paths with `BER_DATASET_DIR`, `BER_TRAIN_DIR`, `BER_TEST_DIR`, `BER_OUTPUT_DIR`, or `BER_ARTIFACT_DIR`.

## Choice C configuration

- `BER_CHOICE_C_ENABLED=1` or `--choice-c`
- `BER_CROSS_ENCODER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `BER_AMBIGUITY_MARGIN=0.08`
- `BER_NEAR_THRESHOLD_WINDOW=0.08`
- `BER_CROSS_ENCODER_BATCH_SIZE=32`
- `BER_CROSS_ENCODER_WEIGHT=0.35`
- `BER_MAX_RERANK_PAIRS=200000`
- `BER_MATCH_THRESHOLD=0.70`

If `sentence-transformers` is unavailable, semantic features are skipped. If Choice C is enabled without it, the pipeline records a fallback and keeps the Choice B LightGBM scores; it never runs the Cross-Encoder over all pairs.

## Validation

Tune the final threshold on a Source-1-level validation split using `evaluation.validation.split_source1_entities`, `filter_candidates_by_source1`, and `evaluation.threshold_tuning.tune_threshold`; never split individual candidate pairs. Run `validate_submission.py` after generating outputs. Final matches are always selected from the generated candidate table.

The command-line `--tune-threshold` path trains on a Source-1-level fold, scores the held-out fold, reports blocking recall and F0.5, and applies the selected threshold to inference. Without that flag, the configured threshold is only a starting value.

## Models and licensing

The default semantic model is `sentence-transformers/all-MiniLM-L6-v2`; the default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Before a scored submission, verify the selected model-card licenses and the competition's allowed-model/parameter rules in the execution environment. No business data or external entity information is fetched by this project. The Python dependencies are standard open-source packages; keep the installed license notices with the environment.

## Tests without dataset

```bash
PYTHONPATH=code python -m unittest discover -s tests -v
python -m compileall -q code validate_submission.py
```
