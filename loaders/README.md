# loaders/

Dataset walking, shared by every detector run. One place decides what counts as an image, which
shards exist, and what ground truth a dataset carries.

## API — `loaders/dataset.py`

| function | returns |
|---|---|
| `iter_dataset(dataset_name, data_dir=DATA_DIR)` | lazily yields `(image_path, label)`; `image_path` is **absolute** |
| `label_for_dataset(dataset_name)` | `0` if the name contains `ffhq` (case-insensitive), else `1` |
| `rel_to_dataset(image_path, dataset_name, data_dir=DATA_DIR)` | the dataset-relative form written to the CSVs, e.g. `shard_01/00042.png` |

| constant | value |
|---|---|
| `IMAGE_EXTENSIONS` | `(".png", ".jpg", ".jpeg")`, matched case-insensitively |
| `DATA_DIR` | `data/`, resolved relative to this file rather than the CWD |

`run.py` imports all three functions. See [`data/README.md`](../data/README.md) for the directory
layout they expect.

## Two invariants worth keeping

**Absolute out, relative in the CSV.** `iter_dataset` yields absolute paths because a detector has
to open the file, while the results CSVs record the relative form — unambiguous next to the
`<dataset_name>` already in the CSV's own filename, and stable across machines. `rel_to_dataset`
is the one conversion between them. Keep that division rather than yielding relative paths and
reconstructing absolute ones per detector.

**`label_for_dataset` is the single definition of ground truth,** since the prediction CSVs
deliberately carry no label column. `run.py`, `auroc_report.py`, `ensemble.py` and
`failure_correlation.py` all import it; a script that re-derives the rule will eventually disagree
with the rest of the project about what is real.

## Implementation constraints

These are properties the current loader has and a replacement would need:

- **Lazy.** Yield paths rather than materializing a list — runs happen at GPU-cluster scale, and
  nothing should assume a dataset fits in memory.
- **Tolerant of partial data.** Shard counts vary by dataset, so discover shards by the `shard_`
  prefix and iterate only what is on disk rather than assuming a count.
- **Deterministic order.** Shards and the files within them are sorted, so two runs over the same
  tree produce rows in the same order. (The analysis scripts still join on `filepath` rather than
  on position, since a detector that errors on an image contributes no row for it.)
