# Deepfake_Detection_Project

Benchmarks pretrained deepfake-detection models against shards of the TrueFake dataset, producing
per-model results CSVs for later correlation analysis (used to pick low-correlation models for
ensembling / mixture of experts).

## Layout

- [`data/`](data/README.md) — evaluation image shards (git-ignored, uploaded incrementally)
- [`models/`](models/README.md) — pretrained detectors under test, each with its own isolated environment
- [`loaders/`](loaders/README.md) — dataset loading code shared across model runs
- [`results/`](results/README.md) — per-model, per-dataset CSV outputs

## Workflow

1. Drop a dataset's shards into `data/<dataset_name>/shard_01..shard_10/`.
2. Set up a model under `models/<model_name>/` with its own environment.
3. Run inference: for each image, load via `loaders/`, score via the model, write a row to
   `results/<model_name>/<dataset_name>.csv`.
4. Once enough models have been run, correlate failure patterns across their results CSVs to pick
   candidates for ensembling / mixture of experts.
