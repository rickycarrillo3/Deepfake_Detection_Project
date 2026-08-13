# Deepfake_Detection_Project

Benchmarks pretrained deepfake-detection models against shards of the TrueFake dataset, producing
per-model results CSVs for later correlation analysis — used to pick low-correlation models as
candidates for ensembling / mixture of experts.

**Fifteen detectors** are registered today: nine under [`models/`](models/README.md) (original
authors' checkpoints) and the six TrueFake-paper baselines under
[`baselines/`](baselines/README.md) (checkpoints from that paper's authors, `tf-` prefixed). All
of them are driven by the single top-level `run.py`.

## Pipeline

```
data/            loaders/               models/<name>/          results/
raw image  --->  yield (image_path,  -> model.py: load() /  ->  <dataset>_<model>_<family>
shards           label) lazily          predict() -> score       _predictions.csv
                                            ^
                                            |
                                     run.py drives all of it
                                     (timing -> results/run_log.csv)
```

## Quick start

Each model owns an isolated environment, so set one up and then invoke the shared runner with
**that model's interpreter**:

```bash
cd models/freqnet && bash setup.sh && cd ../..
models/freqnet/venv/bin/python run.py --model freqnet --dataset ffhq_test
```

Sweeps re-exec each model under its own venv, so launch those with a plain interpreter that has
`pandas` + `tqdm` — that is all `run.py`'s own orchestration needs:

```bash
python3 run.py --list                                        # every registered model
python3 run.py --models tf-npr,freqnet --dataset ffhq_test   # sweep a subset
```

`--limit 10` scores only the first ten images, which is the fastest way to confirm a new model or
dataset is wired up correctly.

## Repository layout

```
run.py                  the one runner for every model (MODELS registry lives here)
metrics.py              THRESHOLD, accuracy, auroc, console summary, run-log append
loaders/                shared dataset walking
data/                   image shards (git-ignored)
models/                 nine detectors, original-author checkpoints
baselines/              six TrueFake-paper detectors, that paper's checkpoints
results/                prediction CSVs + run_log.csv (contents git-ignored)
auroc_report.py         pooled real-vs-fake AUROC across finished CSVs
ensemble.py             ensemble experiments over finished CSVs
failure_correlation.py  per-dataset failure-correlation plots and matrices
```

### `data/`

Git-ignored image shards — too large to track, uploaded incrementally.

```
data/<dataset_name>/shard_01/*.png
data/<dataset_name>/shard_02/*.png
...
```

Each shard holds 1000 PNGs. The shard count **varies by dataset** (FFHQ has 20, ~20,000 images)
and a dataset's shards need not all be present at once, so nothing may hardcode a count. Ground
truth is derived from the folder name: contains `ffhq` (case-insensitive) → `0` (real); anything
else → `1` (fake). See [`data/README.md`](data/README.md).

### `loaders/`

Shared dataset-walking code, so no model reimplements it. `loaders/dataset.py` exposes
`iter_dataset(dataset_name)` — lazily yields absolute `(image_path, label)` pairs, discovering
shards by their `shard_` prefix so partial uploads are fine — plus `label_for_dataset()` (the
`ffhq` rule) and `rel_to_dataset()` (the dataset-relative form written to the CSVs). See
[`loaders/README.md`](loaders/README.md).

### `models/`

One subdirectory per pretrained detector, each with its **own** environment
(`requirements.txt`/`environment.yml`, `venv/`, `weights/`, vendored upstream clone) — different
upstream repos pin conflicting torch/tf versions, so there is intentionally no shared top-level
dependency file. Directories are not 1:1 with detectors:
`models/DMimageDetection/` holds two (`DMimageDetection_DM`, `DMimageDetection_GAN`) sharing one
venv and one weights download. See [`models/README.md`](models/README.md).

### `baselines/`

The six detectors benchmarked by the TrueFake paper (Dell'Anna et al., IJCNN 2025), using
checkpoints supplied by its authors. Same `model.py` contract and same per-directory isolation, so
`run.py` drives them identically; they live in their own tree because they are one coherent set —
all six retrained by those authors on the *same* split, which is what makes their columns
comparable to each other and **not** interchangeable with the same architectures under `models/`.
Hence the `tf-` prefix on every registry name and output CSV. See
[`baselines/README.md`](baselines/README.md).

### `results/`

One flat directory, one CSV per run:
`results/<dataset_name>_<model_name>_<generation_family>_predictions.csv`. Contents are
git-ignored (only `.gitkeep` + README tracked). See [`results/README.md`](results/README.md).

## The runner

There is deliberately **no per-model `run.py`**: they were near-identical copies that drifted
apart and had to be bug-fixed one by one. Every model — `aeroblade` included — goes through the
top-level runner, which needs only `pandas` + `tqdm` (present in every model venv).

| flag | effect |
|---|---|
| `--model NAME` | which registered model to run |
| `--dataset NAME` | `dataset_name` under `data/` |
| `--limit N` | score at most N images (quick test) |
| `--force-label 0\|1` | override the ground-truth label (e.g. a fake-only dataset) |
| `--model-arg KEY=VALUE` | repeatable; forwarded to that model's `load()` |
| `--resize N` | feed every model an identical N×N bicubic copy before its own preprocessing |
| `--all` | sweep every model (and known variant) against one dataset |
| `--baselines` | the same sweep, restricted to the six `tf-*` entries |
| `--models a,b,c` | the same sweep, restricted to a named subset (repeatable) |
| `--keep-repo` / `--keep-venv` | with a sweep, skip deleting clones / venvs afterwards |
| `--dry-run` | with a sweep, print what would run and what would be deleted |
| `--list` | print the registry and exit |

`--model-arg` is how model-specific config travels (`--model-arg train_dataset=sd-v1_4` for
co-spy) without the runner needing to know what it means.

### Sweeps

`--all`, `--baselines` and `--models` are alternatives — pass at most one. Each re-execs
`run.py --model ...` as a fresh subprocess under **that model's own venv**, then deletes that
model's clone and venv to reclaim disk. All three go through `run_all(only=...)`, so cleanup of
*shared* clones and venvs is limited to the models that actually ran and a subset sweep never
deletes a directory it never touched. `--models` names are validated up front, so a typo fails
before the sweep starts rather than hours into it.

Models with more than one meaningfully-different checkpoint are expanded by the runner's
`VARIANT_ARGS`, so `--all` currently means 21 runs across the 15 registry entries:

| model | variants |
|---|---|
| `co-spy` | `train_dataset=progan`, `sd-v1_4` |
| `aide` | `checkpoint=genimage`, `progan`, `sd14` |
| `rine` | `ncls=ldm`, `1`, `2`, `4` |

Launch a sweep with a plain `python3` (pandas + tqdm), **not** one of the models' own venvs — with
venv deletion on, `--all` would otherwise try to delete the venv out from under the interpreter
running it. That case is detected and skipped rather than crashed, but that venv then goes
uncleaned.

### `--resize`

Off by default: each model sees the image at native resolution and applies only its own built-in
resizing. `--resize N` standardizes the input for a controlled cross-model comparison, writing an
N×N (bicubic) copy of each image and handing *that* to `predict()` — before the model's own
preprocessing, without editing any `model.py`. The resampling is fixed to bicubic so the
standardization itself is identical across models. Every CSV records the regime in its `resize`
column so runs are not silently mixed. The resize happens outside the timed span (it is runner
bookkeeping, not inference). `aeroblade` rejects `--resize` (`SUPPORTS_RESIZE = False`).

See [`ResizingRemarks.md`](ResizingRemarks.md) for what each detector does to an image before the
network sees it, and why the native-resolution ones are the slow ones.

## Model contract

Each `model.py` must expose, at **module level** — a class alone is not enough, since this is what
`run.py` calls:

- `load(**kwargs)` — load weights and prepare for inference
- `predict(image_path) -> score` — a float in `[0, 1]`, the predicted fake-probability

Implementing the detector as a class is fine; add a thin module-level shim delegating to it (see
the bottom of `models/freqnet/model.py`). `model.py` must resolve its weights and vendored code
relative to `__file__`, never the CWD — `run.py` imports it from the repo root.

Optional module-level names, read by `run.py` with `getattr` defaults:

| name | default | effect |
|---|---|---|
| `result_stem()` | `"<model_name>"` | the `<model>_<family>` middle of the output filename |
| `SCORE_COLUMN` | `"score"` | renames the score column |
| `SCORE_PRECISION` | `4` | decimals the score and extra columns are rounded to |
| `PREDICTED_LABEL` | `True` | `False` drops `Predicted label` and the console accuracy |
| `SUPPORTS_RESIZE` | `True` | `False` makes `--resize` an error for this model |
| `extra_columns(image_path)` | — | dict of extra columns, written right after the score |

Only `aeroblade` uses the schema hooks; they are for scores that are genuinely not
fake-probabilities, not a general escape hatch.

## Results

### Naming

```
results/<dataset_name>_<model_name>_<generation_family>_predictions.csv
```

The `<model>_<family>` middle comes from `result_stem()` — the generation family the detector was
trained on plus, for checkpoint-varying detectors, whatever keeps their runs from colliding:

| model | stems |
|---|---|
| freqnet, grag2021, cnndetection | `freqnet_GAN`, `grag2021_GAN`, `cnndetection_GAN` |
| DMimageDetection | `DMimageDetection_GAN`, `DMimageDetection_DM` (the registry names already carry the family) |
| co-spy | `co-spy_GAN` (progan), `co-spy_DM` (any other checkpoint) |
| aide | `aide_GAN` (progan), `aide_DM` (sd14), `aide_genimage` (kept literal — a mixed-generator set) |
| rine | `rine_DM` (ldm), `rine1_GAN` / `rine2_GAN` / `rine4_GAN` (the ProGAN-trained N-class checkpoints) |
| aeroblade | `aeroblade` — no family; training-free, so there is none |
| the six baselines | `tf-<name>_GANDM`, e.g. `tf-npr_GANDM` — all six were retrained on one split of **both** families (StyleGAN2 + SDXL) |

Every registered model defines `result_stem()` today; the runner's fallback to a bare
`<model_name>` exists for one that doesn't. Full examples:

```
ffhq_test_freqnet_GAN_predictions.csv
flux_test_DMimageDetection_DM_predictions.csv
ffhq_test_rine1_GAN_predictions.csv
ffhq_test_tf-npr_GANDM_predictions.csv
flux_test_aeroblade_predictions.csv
```

### Schema

| column | type | description |
|---|---|---|
| `filepath` | string | dataset-relative, e.g. `shard_01/00042.png` |
| `score` | float `[0, 1]` | predicted fake-probability |
| `Predicted label` | int `0`/`1` | `score` thresholded at `metrics.THRESHOLD` |
| `resize` | string | `"<N>x<N>"` when the run passed `--resize N`, else `"none"` |

The **ground-truth** label is deliberately *not* a column: it is recoverable from the dataset name
in the filename, and is used only to score accuracy on the console.

`aeroblade` is the one departure — training-free, returning a raw LPIPS reconstruction distance
(negative, larger meaning easier to reconstruct) with no real/fake cutoff defined anywhere in the
paper. It writes `filepath`, `reconstruction score`, one raw-distance column per autoencoder
(`stable-diffusion-v1-1`, `kandinsky-2-1`), and `resize` — no `Predicted label` and no console
accuracy. Do not "restore" either: thresholding an uncalibrated distance at `0.5` would invent a
cutoff the method does not have.

### Timing

Every run appends one row to `results/run_log.csv` (`timestamp`, `model`, `dataset`, `n_images`,
`predict_seconds`, `wall_seconds`) via `metrics.append_run_log()`. `predict_seconds` is the
model's `predict()` calls summed alone; `wall_seconds` is the whole invocation, weight loading
included. There is no `_summary.csv`, and durations stay out of the predictions CSVs, whose schema
is strictly per-image.

Rows are comparable only when hardware, `--resize` regime and dataset matched — the log records
none of the first two, so do not read the file as a cross-model benchmark table without checking.

Accuracy (at `metrics.THRESHOLD`) and prediction time are printed to the **console** at the end of
a run by `metrics.py`, called from the one place so threshold and formatting cannot drift.

## Analysis over finished CSVs

All three read the flat `results/` directory and depend only on pandas (+ numpy/matplotlib), so
they run under any model venv or a plain interpreter:

| script | what it does | output |
|---|---|---|
| `auroc_report.py` | pairs each fake dataset's CSV with the ffhq CSV *for the same model*, pools the scores, reports AUROC + accuracy + per-side rates | console, `results/auroc_summary.csv` |
| `ensemble.py` | combines per-model scores (`mean`, `logodds`, `hard_or`; `--norm raw/rank/zlogit`) and reports whether the ensemble beats its members | console, optional `--out DIR` |
| `failure_correlation.py` | per-dataset failure correlations and oracle accuracy across models | `results/plots/<dataset>/` |

AUROC needs both classes, and every single run covers one single-class dataset — which is why it
lives in `auroc_report.py` over finished CSVs rather than in a run's console summary. Mismatched
`--resize` regimes are skipped with a notice rather than silently pooled. `aeroblade` is out of
scope for `ensemble.py`: its schema carries no `[0, 1]` probability to combine.

Two caveats worth knowing before trusting their output:

> `summarize_results.py` is **stale** — it expects the old `results/<model>/<dataset>.csv` layout
> and a `label` column that `run.py` no longer writes, so it matches nothing today.

> `failure_correlation.py` derives ground truth from its own `REAL_DATASETS = {"ffhq"}`, matched
> **exactly** against the dataset display name, rather than from `loaders.label_for_dataset`'s
> **substring** rule that everything else uses. They agree on `ffhq_faces` but not on e.g.
> `ffhq_test`, which the shared rule calls real and this script would treat as fake.

## Adding a model

1. Write `models/<model_name>/model.py` exposing `load()` / `predict()` (plus `result_stem()` if
   the detector has a generation family or multiple checkpoints).
2. Write `models/<model_name>/setup.sh` — clone upstream, build the venv, fetch weights. Build the
   venv with an overridable interpreter (`PYTHON="${PYTHON:-python3}"`) so a cluster's
   module-loaded Python can be used: `module load python/3.12.12 && PYTHON=python3.12 ./setup.sh`.
   Do not hardcode an interpreter.
3. Add one line to the `MODELS` registry in the top-level `run.py`.

That's it — do **not** write a per-model `run.py`.

## Conventions to preserve

- `data/`, `results/` contents, model weights (`*.pth`, `*.pt`, `*.onnx`, `*.h5`, `*.ckpt`,
  `*.safetensors`), vendored upstream clones and per-model `venv/`/`.venv/`/`env/` directories are
  git-ignored — do not try to commit them.
- Keep environments isolated per-directory; no shared top-level requirements file.
- Results are git-ignored because everything produced so far has been throwaway test output;
  revisit once real benchmark runs land. `run_log.csv` is ignored with them — an append-only file
  committed from several machines would conflict constantly.
