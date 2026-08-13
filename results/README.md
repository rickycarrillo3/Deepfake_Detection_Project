# results/

Holds CSV outputs from model evaluation runs, later used for correlation analysis across models'
failure patterns (to select low-correlation candidates for ensembling / mixture of experts).

## Naming convention

All results live in this one flat directory (no per-model subdirectories), one CSV per run:

```
results/<dataset_name>_<model_name>_<generation_architecture>_predictions.csv
```

The `<model_name>_<generation_architecture>` middle is the model's `result_stem()` (see
`run.py`): the generation family the detector was trained on. Every registered model defines one
today; the runner's fallback to a bare `<model_name>` (no family) exists for one that doesn't.

| model | stem(s) | notes |
|---|---|---|
| `freqnet`, `grag2021`, `cnndetection` | `freqnet_GAN`, `grag2021_GAN`, `cnndetection_GAN` | fixed family |
| `DMimageDetection_GAN` / `_DM` | same as the registry name | the name already carries the family |
| `co-spy` | `co-spy_GAN`, `co-spy_DM` | follows `--model-arg train_dataset` (progan → GAN, anything else → DM) |
| `aide` | `aide_GAN`, `aide_DM`, `aide_genimage` | follows `--model-arg checkpoint` (progan → GAN, sd14 → DM, genimage kept literal — it is a mixed-generator training set) |
| `rine` | `rine_DM`, `rine1_GAN`, `rine2_GAN`, `rine4_GAN` | follows `--model-arg ncls`: the `ldm` checkpoint is diffusion-trained, the 1/2/4-class ones are ProGAN-trained, and their `ncls` sits in the model slot so the four don't overwrite each other |
| `aeroblade` | `aeroblade` | no family suffix — training-free, so there is none it was trained on |
| the six `tf-*` baselines | `tf-npr_GANDM`, `tf-clipd_GANDM`, … | all six were retrained by the TrueFake authors on one split spanning **both** families (StyleGAN2 + SDXL) |

Examples:

```
ffhq_test_freqnet_GAN_predictions.csv
flux_test_DMimageDetection_DM_predictions.csv
ffhq_test_rine1_GAN_predictions.csv
ffhq_test_tf-npr_GANDM_predictions.csv
flux_test_aeroblade_predictions.csv
```

For the checkpoint-varying models the family is what keeps runs from colliding, so it is
load-bearing in the filename rather than decoration: co-spy's two checkpoints land in
`co-spy_GAN` and `co-spy_DM`, and the checkpoint name itself (`sd-v1_4`) never appears.

## CSV schema

| column            | type          | description                                                        |
|-------------------|---------------|--------------------------------------------------------------------|
| `filepath`        | string        | dataset-relative path, e.g. `shard_01/00042.png`                   |
| `score`           | float [0, 1]  | model's predicted fake-probability score                          |
| `Predicted label` | int (0 or 1)  | `score` thresholded at `metrics.THRESHOLD` (0 = real, 1 = fake)    |
| `resize`          | string        | input regime: `"<N>x<N>"` when the run passed `--resize N`, else `"none"` |

### aeroblade's schema

`aeroblade` is the one model that does not output a fake-probability — it is training-free and
returns the LPIPS reconstruction error of the best-matching autoencoder — so it declares the
optional schema hooks in `run.py` (see `models/README.md`) and writes instead:

```
filepath,reconstruction score,stable-diffusion-v1-1,kandinsky-2-1,resize
```

`reconstruction score` is that raw distance (negative LPIPS; larger = easier to reconstruct =
more likely diffusion-generated), **not** a `[0, 1]` probability, so there is no
`Predicted label` column and no accuracy on the console — the method defines no real/fake
cutoff, only a ranking. The two middle columns are the per-autoencoder raw distances, kept for
the correlation analysis. `--resize` is rejected for this model.

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

Written by `metrics.append_run_log()` from the one shared runner, so every model's runs — aeroblade
included, under the name `aeroblade` — land in this log. A sweep (`--all` / `--baselines` / `--models`)
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

## What else lands here

The predictions CSVs and `run_log.csv` are what `run.py` writes. The analysis scripts at the repo
root read those back and write their own output into this directory too:

| path | written by | contents |
|---|---|---|
| `results/auroc_summary.csv` | `auroc_report.py` | one row per (real dataset, fake dataset, model): `auroc`, `accuracy`, `real_acc`, `fake_acc`, `n_real`, `n_fake`, `resize`. `--no-csv` prints only; `--out` redirects it |
| `results/plots/<dataset>/` | `failure_correlation.py` | `combined_vectors.csv` plus the per-dataset correlation matrices and their plots |
| `<dir>/<dataset>_ensemble.csv` | `ensemble.py --out DIR` | per-image combined scores; nothing is written without `--out` |

All three are git-ignored along with everything else here.

## Git

The contents of `results/` are **git-ignored** — only `.gitkeep` files and this README are tracked.
Everything produced so far has been throwaway test output. Revisit once real benchmark runs land.

`run_log.csv` is git-ignored with the rest: it is a record of what a given machine has run, and an
append-only file committed from several machines would conflict on nearly every run. Track it (one
`!/results/run_log.csv` line in `.gitignore`) only if the history is wanted in the repo.
