# data/

Holds the image shards used for model evaluation. This folder is git-ignored — data is uploaded
incrementally and is too large to track in git.

## Layout

```
data/<dataset_name>/shard_01/*.png
data/<dataset_name>/shard_02/*.png
...
data/<dataset_name>/shard_10/*.png
```

Each `shard_XX` subfolder contains 1000 PNG images. A `<dataset_name>` folder may not have all 10
shards present at any given time — loaders should tolerate partial uploads.

## Label rule

Ground-truth label is assigned per `<dataset_name>` folder, based on its name:

- Folder name contains `ffhq` (case-insensitive) → label `0` (real)
- Any other folder name → label `1` (fake)
