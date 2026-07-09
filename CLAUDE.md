# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is currently a scaffold: only directory structure and README files exist, no
loader/model/runner code has been implemented yet. Each subdirectory's README (linked below)
documents the intended contract for code that will live there — read it before implementing
anything in that directory.

## Purpose

Benchmarks pretrained deepfake-detection models against shards of the TrueFake dataset, producing
per-model results CSVs for later correlation analysis (used to pick low-correlation models for
ensembling / mixture of experts).

## Architecture

Data flows: `data/` (raw image shards) → `loaders/` (yield `(image_path, label)` pairs) →
`models/<model_name>/` (score each image) → `results/<model_name>/<dataset_name>.csv`.

- **`data/<dataset_name>/shard_01..shard_10/`** — git-ignored image shards, uploaded incrementally.
  A dataset folder may have only some shards present at any time. Ground-truth label is derived
  from the folder name: contains `ffhq` (case-insensitive) → label `0` (real); anything else →
  label `1` (fake).

- **`loaders/`** — shared dataset walking code. Must yield `(image_path, label)` lazily (never
  load all images into memory — runs happen at GPU-cluster scale), and must tolerate partial
  shard uploads by only iterating what currently exists on disk. Yielded `image_path` must be a
  full/absolute path (needed to open the file). The dataset-relative form written to `results/`
  CSVs (e.g. `shard_01/00042.png`) is derived from it via `rel_to_dataset()`.

- **`models/<model_name>/`** — one subdirectory per pretrained detector, each with its own isolated
  environment (`requirements.txt` or `environment.yml`, `venv/`/`.venv/`, `weights/`) since
  different vendored model repos commonly pin conflicting torch/tf versions. There is intentionally
  no shared top-level dependency file. Once implemented, each model directory should expose:
  - `load()` — load weights and prepare the model for inference
  - `predict(image_path) -> score` — return a float score in `[0, 1]`

- **`results/<model_name>/<dataset_name>.csv`** — output schema is `filepath` (string,
  dataset-relative, e.g. `shard_01/00042.png`; unambiguous alongside the CSV's own
  `<dataset_name>` in its path), `score` (float `[0, 1]`, predicted fake-probability), `label`
  (int, ground truth). Each run
  should also record average per-image processing time (e.g. a companion
  `<dataset_name>_summary.csv` or a run-log row) — mechanism is left to the runner script, but keep
  it consistent across models. Unlike `data/` and model weights, these CSVs are small and tracked
  in git.

## Conventions to preserve when implementing

- `data/` and model weight files (`*.pth`, `*.pt`, `*.onnx`, `*.h5`, `*.ckpt`, `*.safetensors`) and
  per-model `venv/`/`.venv/`/`env/` directories are git-ignored — do not try to commit these.
- Keep model environments isolated per-directory; do not introduce a shared top-level
  requirements file that would force version alignment across models.
