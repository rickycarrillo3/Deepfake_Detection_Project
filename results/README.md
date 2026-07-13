# results/

Holds CSV outputs from model evaluation runs, later used for correlation analysis across models'
failure patterns (to select low-correlation candidates for ensembling / mixture of experts).

## Naming convention

```
results/<model_name>/<dataset_name>.csv
```

## CSV schema

| column       | type          | description                              |
|--------------|---------------|-------------------------------------------|
| `filepath` | string        | dataset-relative path, e.g. `shard_01/00042.png` |
| `score`      | float [0, 1]  | model's predicted fake-probability score  |
| `label`      | int (0 or 1)  | ground-truth label (0 = real, 1 = fake)   |

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
