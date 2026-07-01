# loaders/

Holds dataset loading code shared across model runs.

## Responsibility

A loader walks a `data/<dataset_name>/shard_XX/` tree (see [`data/README.md`](../data/README.md))
and yields `(image_path, label)` pairs, where `label` is determined by the
`ffhq`-in-folder-name rule.

## Conventions

- **Lazy iteration** — yield paths rather than loading all images into memory, since runs happen
  at GPU-cluster scale.
- **Partial data tolerant** — not all 10 shards may be present yet; loaders should only iterate
  over what currently exists on disk.
- **Full paths** — yielded `image_path` values should be full/absolute paths, matching the
  `image_path` column expected in `results/` CSVs.
