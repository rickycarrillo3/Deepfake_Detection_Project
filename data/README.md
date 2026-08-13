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

This rule lives in `loaders.label_for_dataset()` and is the only source of ground truth in the
repo — the prediction CSVs deliberately carry no label column (see
[`results/README.md`](../results/README.md)). `run.py --force-label 0|1` overrides it for a run
whose folder name doesn't follow the convention.

So **the dataset folder name is load-bearing**: it decides the label, and it becomes the first
part of every output CSV's filename. Name a new dataset before scoring it, not after.

## Reading a dataset

Never walk `data/` directly from a `model.py` — go through
[`loaders/`](../loaders/README.md), which handles the shard discovery, the label rule and the
absolute-vs-relative path split in one place:

```python
from loaders.dataset import iter_dataset

for image_path, label in iter_dataset("ffhq_test"):
    ...
```
