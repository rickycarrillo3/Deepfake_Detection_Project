# Repository Structure

This project benchmarks pretrained deepfake-detection models against shards of image datasets
(e.g. TrueFake), producing per-model results CSVs used for correlation analysis — picking
low-correlation models as candidates for ensembling / mixture of experts.

## Pipeline

```
data/  --->  loaders/  --->  models/<model_name>/  --->  results/<dataset_name>_<model_name>_<family>_predictions.csv
(raw shards)  (yield        (score each image,          (one flat directory, one CSV per run)
              (image_path,   driven by the top-level
               label) pairs)  run.py)
```

## Folders

### `data/`

Evaluation image shards. Git-ignored — too large to track, uploaded incrementally.

```
data/<dataset_name>/shard_01/*.png
data/<dataset_name>/shard_02/*.png
...
data/<dataset_name>/shard_NN/*.png
```

Each shard holds 1000 PNGs. The shard count varies by dataset — FFHQ has 20 (~20,000 images) —
and a dataset's shards need not all be present at once, so nothing hardcodes a count. Ground-truth
label is derived from the folder name: contains `ffhq` (case-insensitive) → real (`0`); anything
else → fake (`1`).

### `loaders/`

Dataset-loading code shared across every model run, so each model doesn't reimplement shard
walking. `loaders/dataset.py` exposes `iter_dataset(dataset_name)`, which walks
`data/<dataset_name>/shard_XX/`, applies the `ffhq` label rule, and lazily yields absolute
`(image_path, label)` pairs — tolerating shards that haven't been uploaded yet.

### `models/`

One subdirectory per pretrained detector under test, e.g. `models/co-spy/`. Each model directory
manages its **own environment** (`requirements.txt`, `venv/`, `weights/`, plus a vendored copy of
the model's GitHub repo) since different detectors commonly pin conflicting torch/tf versions —
there's intentionally no shared top-level dependency file.

Every model directory exposes the same two-function contract for the runner to call:

- `load()` — load weights, prepare the model for inference
- `predict(image_path) -> score` — float in `[0, 1]`, predicted fake-probability

`models/co-spy/` is the first implementation of this contract (wraps `Megum1/CO-SPY`) and can be
used as a template for porting additional models — see its `README.md` and `run.py` for the
`loaders/` + `model.py` + results-writing pattern to follow.

### `results/`

CSV outputs from model evaluation runs. All of them live in **one flat directory** — no per-model
or per-dataset subdirectories, because the filename already carries the dataset, the model, and
the generation family the detector was trained on:

```
results/<dataset_name>_<model_name>_<generation_family>_predictions.csv
```

CSV schema: `filepath` (dataset-relative, e.g. `shard_01/00042.png`), `score` (float `[0, 1]`,
predicted fake-probability), `Predicted label` (int, `score` thresholded at `metrics.THRESHOLD`),
and `resize` (the `--resize` regime that produced the run, else `"none"`). Ground truth is *not* a
column — it is recoverable from the dataset name in the filename.

There is no `_summary.csv`. Per-run timing goes to `results/run_log.csv`, one appended row per
run; accuracy and AUROC are printed to the console. See `results/README.md` for both in full.

Contents are git-ignored (only `.gitkeep` and `README.md` are tracked) — everything produced so
far is throwaway test output.

## Adding a new model

1. Create `models/<model_name>/` with its own isolated environment and vendored repo.
2. Implement `model.py` exposing `load()` / `predict(image_path) -> score`.
3. Add one line for it to the `MODELS` registry in the **top-level** `run.py`, which walks the
   dataset, scores it, and writes the flat `results/` CSV. Do *not* add a per-model `run.py`.
4. Once several models have results, correlate their failure patterns to pick low-correlation
   candidates for ensembling.
