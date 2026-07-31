# Repository Structure

This project benchmarks pretrained deepfake-detection models against shards of image datasets
(e.g. TrueFake), producing per-model results CSVs used for correlation analysis — picking
low-correlation models as candidates for ensembling / mixture of experts.

## Pipeline

```
data/  --->  loaders/  --->  models/<model_name>/  --->  results/<model_name>/<dataset_name>.csv
(raw shards)  (yield        (score each image)          (score + ground truth per image)
              (image_path, label) pairs)
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

CSV outputs from model evaluation runs. Small and tracked in git (unlike `data/` and weights).

```
results/<model_name>/<dataset_name>.csv
results/<model_name>/<dataset_name>_summary.csv   # avg per-image processing time
```

CSV schema: `filepath` (full path to the evaluated image), `score` (float `[0, 1]`), `label`
(ground truth, 0 = real / 1 = fake).

## Adding a new model

1. Create `models/<model_name>/` with its own isolated environment and vendored repo.
2. Implement `model.py` exposing `load()` / `predict(image_path) -> score`.
3. Implement `run.py` that imports `iter_dataset` from `loaders/`, runs predictions over a
   dataset, and writes `results/<model_name>/<dataset_name>.csv` (+ `_summary.csv`).
4. Once several models have results, correlate their failure patterns to pick low-correlation
   candidates for ensembling.
