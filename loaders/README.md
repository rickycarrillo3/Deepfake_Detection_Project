# loaders/

Holds dataset loading code shared across model runs.

## Responsibility

A loader walks a `data/<dataset_name>/shard_XX/` tree (see [`data/README.md`](../data/README.md))
and yields `(filepath, label)` pairs, where `label` is determined by the
`ffhq`-in-folder-name rule.

## Conventions

- **Lazy iteration** — yield paths rather than loading all images into memory, since runs happen
  at GPU-cluster scale.
- **Partial data tolerant** — the shard count varies by dataset (FFHQ has 20) and uploads are
  incremental, so loaders must discover shards by the `shard_` prefix rather than assume a count,
  and only iterate over what currently exists on disk.
- **Full paths** — yielded `filepath` values should be full/absolute paths, matching the
  `image_path` column expected in `results/` CSVs.
