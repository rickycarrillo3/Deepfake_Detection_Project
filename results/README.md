# results/

Holds CSV outputs from model evaluation runs, later used for correlation analysis across models'
failure patterns (to select low-correlation candidates for ensembling / mixture of experts).

## Naming convention

All results live in this one flat directory (no per-model subdirectories), one CSV per run:

```
results/<dataset_name>_<model_name>_<generation_architecture>_predictions.csv
```

The `<model_name>_<generation_architecture>` middle is the model's `result_stem()` (see
`run.py`): the generation family the detector was trained on — `GAN` or `DM` — and, for
checkpoint-varying detectors, whatever distinguishes their checkpoints so runs don't collide
(co-spy's `train_dataset`, aide's `checkpoint`, rine's `ncls` in the model slot). Models without
a `result_stem()` fall back to just `<model_name>` (no family). Examples:

```
ffhq_test_freqnet_GAN_predictions.csv
flux_test_DMimageDetection_DM_predictions.csv
ffhq_test_rine1_DM_predictions.csv
flux_test_co-spy_sd-v1_4_DM_predictions.csv
```

`aeroblade` is the exception: it has its own bespoke runner, emits extra per-autoencoder
distance columns, and gets no family suffix (`<dataset_name>_aeroblade_predictions.csv`).

## CSV schema

| column            | type          | description                                                        |
|-------------------|---------------|--------------------------------------------------------------------|
| `filepath`        | string        | dataset-relative path, e.g. `shard_01/00042.png`                   |
| `score`           | float [0, 1]  | model's predicted fake-probability score                          |
| `Predicted label` | int (0 or 1)  | `score` thresholded at `metrics.THRESHOLD` (0 = real, 1 = fake)    |
| `resize`          | string        | input regime: `"<N>x<N>"` when the run passed `--resize N`, else `"none"` |

The **ground-truth** label is deliberately *not* a column: it is recoverable from the dataset
name in the filename (contains `ffhq` → `0`/real, else → `1`/fake — see
`loaders.label_for_dataset`) and is used only to score accuracy on the console.

## Run timing — `run_log.csv`

Every run appends one row to `results/run_log.csv`, the one file here that accumulates across runs
rather than being rewritten by one:

| column            | type   | description                                                          |
|-------------------|--------|----------------------------------------------------------------------|
| `timestamp`       | string | ISO 8601, local time, seconds resolution                             |
| `model`           | string | the `MODELS` registry name from `run.py` (e.g. `tf-npr`)             |
| `dataset`         | string | `dataset_name` under `data/`                                         |
| `n_images`        | int    | rows written to that run's predictions CSV (skipped images excluded) |
| `predict_seconds` | float  | the model's `predict()` calls summed on their own                    |
| `wall_seconds`    | float  | the whole invocation, weight loading and CSV writing included         |

Written by `metrics.append_run_log()`, which aeroblade's bespoke runner calls too, so its runs
appear in the same log under the name `aeroblade`. A sweep (`--all` / `--baselines` / `--models`)
runs each model as its own `run.py --model ...` subprocess, so it appends one row per model, not
one for the sweep.

The two durations answer different questions. `predict_seconds` is the model: one shared runner
now times that span with the same code for every model, so the numbers are measured uniformly.
`wall_seconds` is operational — how long the job held the machine, for planning cluster time — and
says as much about weight loading and disk as about the detector.

Rows are only comparable to each other when what surrounds them matched: the log records neither
the hardware nor the input resolution (`--resize`), and models batch differently, so compare
within a set of runs you know shared those, not across the file at large.

Timing stays out of the **predictions** CSVs, whose schema is strictly per-image. Keep it that
way — the separate file is what stops a duration from being read as a per-image column.

## Git

The contents of `results/` are **git-ignored** — only `.gitkeep` files and this README are tracked.
Everything produced so far has been throwaway test output. Revisit once real benchmark runs land.

`run_log.csv` is git-ignored with the rest: it is a record of what a given machine has run, and an
append-only file committed from several machines would conflict on nearly every run. Track it (one
`!/results/run_log.csv` line in `.gitignore`) only if the history is wanted in the repo.
