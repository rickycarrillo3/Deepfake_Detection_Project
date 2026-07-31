# data/

Holds the image shards used for model evaluation. This folder is git-ignored — data is uploaded
incrementally and is too large to track in git.

## Layout

```
data/<dataset_name>/shard_01/*.png
data/<dataset_name>/shard_02/*.png
...
data/<dataset_name>/shard_NN/*.png
```

Each `shard_XX` subfolder contains 1000 PNG images. **The number of shards varies by dataset** —
FFHQ has 20 (`shard_01` … `shard_20`, so ~20,000 images); other datasets have their own counts.
Nothing should hardcode a shard count: `loaders/` discovers shards by walking for the `shard_`
prefix, which also means a `<dataset_name>` folder need not have all of its shards present at any
given time. Uploads are incremental, so partial trees are normal and must be tolerated.

## Label rule

Ground-truth label is assigned per `<dataset_name>` folder, based on its name:

- Folder name contains `ffhq` (case-insensitive) → label `0` (real)
- Any other folder name → label `1` (fake)
