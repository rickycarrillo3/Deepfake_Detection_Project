# results/

Every output of the benchmark: one prediction CSV per run, an append-only timing log, and
whatever the analysis scripts write back. The contents are **git-ignored** and reproducible —
this directory ships with only its README.

The per-image scores collected here are the input to the correlation analysis, which is what the
benchmark exists to produce: identifying detectors that fail on *different* images, and are
therefore worth combining.

## Prediction CSVs

All results live in this one flat directory — no per-detector subdirectories — with one CSV per
run:

```
results/<dataset_name>_<model_name>_<generation_family>_predictions.csv
```

### Schema

| column | type | description |
|---|---|---|
| `filepath` | string | dataset-relative path, e.g. `shard_01/00042.png` |
| `score` | float `[0, 1]` | the detector's predicted fake-probability |
| `Predicted label` | int `0`/`1` | `score` thresholded at `metrics.THRESHOLD` (0 = real, 1 = fake) |
| `resize` | string | input regime: `"<N>x<N>"` when the run passed `--resize N`, else `"none"` |

The **ground-truth label is deliberately not a column.** It is recoverable from the dataset name
in the filename — contains `ffhq` → `0`/real, else → `1`/fake, per `loaders.label_for_dataset` —
and keeping one source of truth prevents a per-row copy from disagreeing with the filename. It is
used only to score accuracy on the console.

Because the label comes from the dataset, **every CSV is single-class**. AUROC therefore requires
pooling a real dataset's CSV with a fake one's for the same detector, which is what
`auroc_report.py` does.

### Filename stems

The `<model_name>_<generation_family>` middle is the detector's `result_stem()`: the generation
family it was trained on, plus whatever distinguishes checkpoints of the same detector. For the
checkpoint-varying detectors this is load-bearing rather than decorative — it is what stops two
runs of the same detector from overwriting each other.

| detector | stem(s) | determined by |
|---|---|---|
| `freqnet`, `grag2021`, `cnndetection` | `freqnet_GAN`, `grag2021_GAN`, `cnndetection_GAN` | fixed family |
| `DMimageDetection_GAN` / `_DM` | same as the registry name | the registry name already carries the family |
| `co-spy` | `co-spy_GAN`, `co-spy_DM` | `--model-arg train_dataset` (progan → GAN, anything else → DM); the checkpoint name `sd-v1_4` never appears |
| `aide` | `aide_GAN`, `aide_DM`, `aide_genimage` | `--model-arg checkpoint` (progan → GAN, sd14 → DM, genimage kept literal — a mixed-generator training set) |
| `rine` | `rine_DM`, `rine1_GAN`, `rine2_GAN`, `rine4_GAN` | `--model-arg ncls`: the `ldm` checkpoint is diffusion-trained; the 1/2/4-class ones are ProGAN-trained, with `ncls` in the model slot so the four don't collide |
| `aeroblade` | `aeroblade` | no family — training-free, so there is none it was trained on |
| the six `tf-*` baselines | `tf-npr_GANDM`, `tf-clipd_GANDM`, … | one split spanning **both** families (StyleGAN2 + SDXL) |

```
ffhq_test_freqnet_GAN_predictions.csv
flux_test_DMimageDetection_DM_predictions.csv
ffhq_test_rine1_GAN_predictions.csv
ffhq_test_tf-npr_GANDM_predictions.csv
flux_test_aeroblade_predictions.csv
```

Every registered detector defines a `result_stem()`; the runner's fallback to a bare
`<model_name>` with no family exists for one that doesn't.

### `aeroblade`'s schema

`aeroblade` is the one detector that does not output a fake-probability. It is training-free and
returns the LPIPS reconstruction error of the best-matching autoencoder, so it declares the
optional schema hooks in `run.py` and writes instead:

```
filepath,reconstruction score,stable-diffusion-v1-1,kandinsky-2-1,resize
```

`reconstruction score` is that raw distance — negative LPIPS, larger meaning easier to
reconstruct and so more likely diffusion-generated — and **not** a `[0, 1]` probability. There is
consequently no `Predicted label` column and no console accuracy: the method defines no real/fake
cutoff, only a ranking. The two middle columns are the per-autoencoder raw distances, retained for
the correlation analysis. `--resize` is rejected for this detector.

## Timing — `run_log.csv`

Every run appends one row, making this the one file here that accumulates across runs rather than
being rewritten by one:

| column | type | description |
|---|---|---|
| `timestamp` | string | ISO 8601, local time, seconds resolution |
| `model` | string | the `MODELS` registry name from `run.py` (e.g. `tf-npr`) |
| `dataset` | string | `dataset_name` under `data/` |
| `n_images` | int | rows written to that run's predictions CSV (skipped images excluded) |
| `predict_seconds` | float | the detector's `predict()` calls summed on their own |
| `wall_seconds` | float | the whole invocation, weight loading and CSV writing included |

Written by `metrics.append_run_log()` from the one shared runner, so every detector's runs —
`aeroblade` included — land in the same log. A sweep runs each detector as its own
`run.py --model ...` subprocess, so it appends one row per detector rather than one for the sweep.

The two durations answer different questions. `predict_seconds` characterises the detector, and is
measured by the same code for every one of them, which is what makes it comparable at all.
`wall_seconds` is operational — how long the job held the machine, for planning cluster time — and
says as much about weight loading and disk as about the detector.

> Rows are comparable only when what surrounds them matched. The log records neither the hardware
> nor the input resolution (`--resize`), and detectors batch differently, so compare within a set
> of runs you know shared those rather than reading the file as a cross-detector benchmark table.

Timing stays out of the predictions CSVs, whose schema is strictly per-image; the separate file is
what stops a run-level duration from being read as a per-image column.

## Analysis output

The analysis scripts at the repository root read the prediction CSVs back and write their own
output here:

| path | written by | contents |
|---|---|---|
| `results/auroc_summary.csv` | `auroc_report.py` | one row per (real dataset, fake dataset, detector): `auroc`, `accuracy`, `real_acc`, `fake_acc`, `n_real`, `n_fake`, `resize`. `--no-csv` prints only; `--out` redirects it |
| `results/plots/<dataset>/` | `failure_correlation.py` | `combined_vectors.csv`, the three correlation matrices (`error_pearson.csv`, `failure_pearson.csv`, `failure_spearman.csv`) and their heatmaps, plus `accuracy.png` |
| `<dir>/<dataset>_ensemble.csv` | `ensemble.py --out DIR` | per-image combined scores; nothing is written without `--out` |

## Git

Only this README and a `.gitkeep` are tracked; everything else here is git-ignored. The CSVs are
reproducible from the
code and the dataset, so there is nothing to gain from versioning them, and `run_log.csv` is
append-only — committed from several machines it would conflict on nearly every run. To keep the
timing history in the repository anyway, un-ignore it with a single `!/results/run_log.csv` line
in `.gitignore`.
