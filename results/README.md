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
flux_test_dmimagedetection_DM_predictions.csv
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

## Run timing

None is recorded. The models time themselves differently (some did not time at all before the
runner was consolidated), so per-image or total durations are not comparable across models and are
deliberately not written out as if they were. Images that fail to score are skipped and counted to
the console, not to a CSV.

Do not reintroduce a timing CSV without first making the measurement genuinely comparable across
models.

## Git

The contents of `results/` are **git-ignored** — only `.gitkeep` files and this README are tracked.
Everything produced so far has been throwaway test output. Revisit once real benchmark runs land.
