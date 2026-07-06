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
| `filepath` | string        | full path to the evaluated image          |
| `score`      | float [0, 1]  | model's predicted fake-probability score  |
| `label`      | int (0 or 1)  | ground-truth label (0 = real, 1 = fake)   |

## Run timing

Each run also records the average per-image processing time (e.g. a companion
`results/<model_name>/<dataset_name>_summary.csv`, or a row appended to a run log). The exact
mechanism is left to the runner script; the convention here just keeps it consistent across
models.

Unlike `data/` and model weights, these CSVs are small and are tracked in git.
