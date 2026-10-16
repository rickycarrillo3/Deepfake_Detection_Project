# data/

The image shards every detector is scored against. **Git-ignored** — the images are far too large
to track, so this directory ships empty and you populate it yourself.

## Obtaining the data

The benchmark is built around [TrueFake](https://arxiv.org/abs/2504.20658) (Dell'Anna, Montibeller
& Boato, IJCNN 2025); see the authors' release at
[MMLab-unitn/TrueFake-IJCNN25](https://github.com/MMLab-unitn/TrueFake-IJCNN25) for access.

Nothing in the harness is TrueFake-specific, though. Any directory of images that follows the
layout below can be scored, which is how the real (FFHQ) and fake sides are kept as separate
datasets here.

## Layout

```
data/<dataset_name>/shard_01/<images>
data/<dataset_name>/shard_02/<images>
...
data/<dataset_name>/shard_NN/<images>
```

Three things about this layout are load-bearing:

- **Subdirectories must be named `shard_*`.** The loader discovers them by that prefix; anything
  else is skipped. Images placed directly in `data/<dataset_name>/` are not found.
- **Files are selected by extension** — `.png`, `.jpg`, `.jpeg`, matched case-insensitively
  (`loaders.dataset.IMAGE_EXTENSIONS`). Anything else in a shard is ignored.
- **The dataset directory name decides the label** (see below) *and* becomes the first component
  of every output CSV's filename. Name a dataset before scoring it, not after.

A misconfigured tree therefore yields no images rather than an error, and `run.py` reports only
"No images scored". If you see that, check the shard prefix and the file extensions first.

Shard size is a convention, not a constraint: each `shard_XX` here holds 1000 images, and FFHQ has
20 of them (~20,000 images). **Nothing hardcodes a shard count** — the loader iterates whatever
exists on disk, so a dataset whose shards are only partly present is fine and will simply score
the images that are there.

## Label rule

Ground truth is assigned per dataset directory, from its name:

| directory name | label |
|---|---|
| contains `ffhq` (case-insensitive) | `0` — real |
| anything else | `1` — fake |

Each dataset is therefore single-class by construction. This rule lives in
`loaders.label_for_dataset()` and is the **only** source of ground truth in the repository — the
prediction CSVs deliberately carry no label column (see [`results/README.md`](../results/README.md)).
AUROC consequently needs two datasets pooled, which is what `auroc_report.py` does.

`run.py --force-label 0|1` overrides the rule for a dataset whose name doesn't follow the
convention.

## Reading a dataset

Go through [`loaders/`](../loaders/README.md) rather than walking `data/` directly, so that shard
discovery, the label rule and the absolute-vs-relative path split stay in one place:

```python
from loaders.dataset import iter_dataset

for image_path, label in iter_dataset("ffhq_test"):
    ...
```
