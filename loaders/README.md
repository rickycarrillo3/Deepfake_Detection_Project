# loaders/

Holds dataset loading code shared across model runs.

## Responsibility

A loader walks a `data/<dataset_name>/shard_XX/` tree (see [`data/README.md`](../data/README.md))
and yields `(filepath, label)` pairs, where `label` is determined by the
`ffhq`-in-folder-name rule.

## API — `loaders/dataset.py`

| function | returns |
|---|---|
| `iter_dataset(dataset_name, data_dir=DATA_DIR)` | lazily yields `(image_path, label)`, `image_path` absolute |
| `label_for_dataset(dataset_name)` | `0` if the name contains `ffhq` (case-insensitive), else `1` |
| `rel_to_dataset(image_path, dataset_name, data_dir=DATA_DIR)` | the dataset-relative form written to the CSVs, e.g. `shard_01/00042.png` |

`run.py` imports all three. The split matters: `iter_dataset` hands out **absolute** paths because
a model has to open the file, while the results CSVs record the **relative** form — unambiguous
next to the `<dataset_name>` already in the CSV's own filename, and stable across machines. Keep
that division rather than yielding relative paths and rebuilding them per model.

`label_for_dataset` is meant to be the single definition of ground truth, since the prediction
CSVs deliberately carry no label column. `run.py`, `auroc_report.py` and `ensemble.py` all import
it.

> **Known divergence:** `failure_correlation.py` does *not*. It keeps its own
> `REAL_DATASETS = {"ffhq"}` and matches the dataset's display name **exactly** (after stripping a
> trailing `_faces`), where `label_for_dataset` matches `ffhq` as a **substring**. The two agree
> on `ffhq_faces` but not on a name like `ffhq_test`, which the shared rule calls real and
> `failure_correlation.py` would treat as fake. Point it at `label_for_dataset` before running it
> on any real dataset whose name isn't exactly `ffhq` or `ffhq_faces`.

## Conventions

- **Lazy iteration** — yield paths rather than loading all images into memory, since runs happen
  at GPU-cluster scale.
- **Partial data tolerant** — the shard count varies by dataset (FFHQ has 20) and uploads are
  incremental, so loaders must discover shards by the `shard_` prefix rather than assume a count,
  and only iterate over what currently exists on disk.
- **Full paths** — yielded `filepath` values should be full/absolute paths, matching the
  `image_path` column expected in `results/` CSVs.
