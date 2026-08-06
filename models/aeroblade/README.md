# models/aeroblade/

Wraps [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade)
(CVPR 2024 — "AEROBLADE: Training-Free Detection of Latent Diffusion
Images Using Autoencoder Reconstruction Error") behind this repo's
`load()` / `predict(image_path) -> score` contract, plus a
`predict_batch()` that `run.py` actually uses (see below for why).

Ported from `scripts/run_aeroblade.py`, which calls
`aeroblade.high_level_funcs.compute_distances()`.

## How AEROBLADE is different from the other models here

- **Training-free.** No classifier weights — it reconstructs each image
  with several pretrained LDM autoencoders (SD1, SD2, Kandinsky 2.1 by
  default) and measures LPIPS reconstruction error. Smaller error ⇒
  more likely fake (the AE "recognizes" its own generator's output).
- **Batch-oriented by *directory*, not by image list.** `compute_distances()`'s
  loop is `for dir in dirs: ds = ImageFolder(dir); for repo_id in
  repo_ids: compute_reconstructions(ds, repo_id, ...)` — each directory
  you pass gets one batched reconstruction pass per AE; each individual
  file gets its own. So `run.py` groups images by shard directory and
  calls `model.predict_batch(shard_dir)` once per shard (a whole
  `data/<dataset>/shard_XX/` at a time), not `predict()` per image.
  `predict(image_path)` still exists for contract compatibility, it's
  just not what you want to use at scale.
- **Same-dimension requirement.** AEROBLADE's docs note all images in one
  directory must share dimensions — shards already satisfy this.
- **`compute_max=True` (the default) does the AE reduction for you.**
  `compute_distances()` returns a long-format DataFrame (columns `dir`,
  `image_size`, `repo_id`, `transform`, `distance_metric`, `file`,
  `distance`) and, with `compute_max=True`, adds a `repo_id="max"` row
  per file holding the max (best-AE) distance. `model.py` reads that row
  directly rather than re-deriving it.
- **Score isn't a calibrated probability.** Distances are negative LPIPS
  reconstruction errors; there's no paper-defined real/fake threshold,
  only relative ranking (used for AP/ROC in their eval). `model.py`
  passes the max-AE distance through a sigmoid purely to land in `[0, 1]`
  for the `results/` schema — monotonic only, so it preserves rank and
  correlation with other models' scores, but don't read the absolute
  value as "% fake."

## Setup

```bash
cd models/aeroblade
bash setup.sh
```

`setup.sh` (safe to re-run):
1. Clones the upstream repo to `aeroblade/`.
2. Creates `aeroblade_env`, Python 3.10 (upstream's tested version).
3. Installs `aeroblade/requirements.txt`, then `pip install -e ./aeroblade`
   itself, then this directory's `requirements.txt`.

First run downloads AE weights from HuggingFace (`CompVis/stable-diffusion-v1-1`,
`stabilityai/stable-diffusion-2-base`, `kandinsky-community/kandinsky-2-1`)
— log in with `aeroblade_env/bin/huggingface-cli login` if any require
accepting a license on the Hub.

## Run

```bash
cd models/aeroblade
PYTHONPATH=<repo_root> aeroblade_env/bin/python run.py <dataset_name>
```

Writes `results/<dataset_name>_aeroblade_predictions.csv` — all results
share one flat `results/` directory, and aeroblade takes no generation-family
suffix — with the standard
`filepath`, `score`, `label` columns, **plus one extra column per
autoencoder** holding its raw (unsquashed) reconstruction distance — you
asked to keep these for the correlation analysis, since low agreement
across AEs might itself be informative.
