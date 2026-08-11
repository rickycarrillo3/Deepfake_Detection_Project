# models/aeroblade/

Wraps [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade)
(CVPR 2024 — "AEROBLADE: Training-Free Detection of Latent Diffusion
Images Using Autoencoder Reconstruction Error") behind this repo's
`load()` / `predict(image_path)` contract, so the shared top-level
`run.py` drives it like every other model.

Ported from `scripts/run_aeroblade.py`, which calls
`aeroblade.high_level_funcs.compute_distances()`.

## How AEROBLADE differs from the other models here

- **Training-free.** No classifier weights — it reconstructs each image
  with pretrained LDM autoencoders (SD1 and Kandinsky 2.1 as configured
  in `model.py`; SD2 is commented out) and measures LPIPS reconstruction
  error. Smaller error ⇒ more likely diffusion-generated (the AE
  "recognizes" its own generator's output).
- **Its score is not a probability.** `predict()` returns the raw
  best-AE distance: a *negative* LPIPS error, where larger (less
  negative) means easier to reconstruct. The paper only ever ranks these
  (AP/ROC) — it defines no real/fake cutoff — so the CSV names the
  column `reconstruction score`, carries no `Predicted label`, and no
  accuracy is printed. `compute_distances(..., compute_max=True)` (the
  default) already adds the `repo_id="max"` row per file; `model.py`
  reads it rather than re-deriving it.
- **Batched by *directory*, not by image.** `compute_distances()`'s loop
  is `for dir in dirs: ds = ImageFolder(dir); for repo_id in repo_ids:
  compute_reconstructions(ds, repo_id, ...)` — each directory passed
  gets one batched reconstruction pass per AE, while each individual
  file passed gets its own. So `predict()` keeps the one-image-at-a-time
  contract but **caches by parent directory**: the first image of a
  shard scores the whole shard in one call, and the rest are cache hits.
  Only the current shard is held in memory.
- **No `--resize`.** `model.py` sets `SUPPORTS_RESIZE = False` and
  `run.py` errors out. The resize path rewrites every image to one
  reused temp filename, which the directory cache would read as a single
  one-image directory and hand the first image's scores to the whole run.
- **Same-dimension requirement.** AEROBLADE requires all images in one
  directory to share dimensions — shards already satisfy this.

`model.py` declares four schema hooks (`SCORE_COLUMN`, `SCORE_PRECISION`,
`PREDICTED_LABEL`, `SUPPORTS_RESIZE`) plus `extra_columns()`; see
`models/README.md` for what `run.py` does with them.

## Setup

```bash
cd models/aeroblade
bash setup.sh                       # or: PYTHON=python3.10 ./setup.sh
```

`setup.sh` (safe to re-run):
1. Clones the upstream repo to `aeroblade/`.
2. Creates `venv/` (upstream tests on Python 3.10 — pass `PYTHON=` to pick it).
3. Installs `aeroblade/requirements.txt`, then `pip install -e ./aeroblade`,
   then this directory's `requirements.txt`.

### Interpreter, and the cluster

This model constrains its Python more than the others: upstream pins
`torch==2.1.2`, whose wheels stop at cp311, so **Python 3.12+ cannot install
this stack at all**. The cluster's default `python3` is 3.12, so `setup.sh`
handles it rather than leaving it as a manual step — when the default
interpreter is too new and conda/mamba is available, it creates `venv/` *as* a
Python 3.10 conda env. Same path, same layout, so `run.py` finds it exactly as
it finds every other model's venv:

```bash
module load miniforge
cd models/aeroblade && ./setup.sh          # prints: venv/ interpreter: Python 3.10.x
```

Precedence, in order:

1. `PYTHON=` is always honoured, per the repo convention (`PYTHON=python3.10
   ./setup.sh`). If it points at 3.12+, setup warns that pip is about to fail
   and continues anyway — your override, your call.
2. Otherwise, if the default `python3` is < 3.12, a plain `python -m venv`.
3. Otherwise, conda (found via `$CONDA_EXE`, `conda`, or `mamba` — module
   systems often expose conda as a shell function a script cannot see, but
   still export `CONDA_EXE`) creates `venv/` at Python 3.10.
4. Otherwise, it exits with an explanation rather than building a venv that
   cannot install the requirements.

Keeping upstream's pinned torch/diffusers is deliberate: they are the versions
the paper's numbers came from. Bumping torch for cp312 would drag diffusers
along, and the reconstruction distances would need re-validating before a run
could be trusted.

First run downloads AE weights from HuggingFace
(`CompVis/stable-diffusion-v1-1`, `kandinsky-community/kandinsky-2-1`) —
log in with `venv/bin/huggingface-cli login` if any require accepting a
license on the Hub.

## Run

```bash
models/aeroblade/venv/bin/python run.py --model aeroblade --dataset <dataset_name>
```

Optional `--model-arg`s, all forwarded to `load()`:

| arg | default | meaning |
|---|---|---|
| `device` | auto | `cpu` forces CPU (by hiding the GPUs); otherwise `cuda` when available |
| `batch_size` | 4 | reconstruction batch size — raise it to fill a bigger GPU |
| `num_workers` | 2 | dataloader workers |

`load()` prints the resolved device, batch size and worker count at the
top of every run.

Writes `results/<dataset_name>_aeroblade_predictions.csv`:

```
filepath,reconstruction score,stable-diffusion-v1-1,kandinsky-2-1,resize
```

The two middle columns are each AE's own raw distance, kept for the
correlation analysis since disagreement between AEs may itself be
informative. No family suffix in the filename: a training-free method
was not trained on a generation family.

## GPU / CPU

Upstream resolves the device itself (`aeroblade.misc.device()` →
`"cuda"` if `torch.cuda.is_available()`), so on a GPU box the code path
is upstream's own: fp16 pipelines, `enable_model_cpu_offload()` and
`torch.compile`. All three fail on a CPU-only machine, so `load()`
installs a shim (`_patch_for_cpu()` in `model.py`) that loads in fp32,
neutralizes the offload call and makes `torch.compile` a passthrough.
That shim is **inert whenever CUDA is present**, so it cannot change GPU
results. One patch does apply on both: `safe_mkdir()`'s interactive
"directory exists, continue? (y/n)" prompt is answered yes
automatically, since a batch job has no stdin (it is only reached when a
cached reconstruction directory is incomplete, e.g. after an
interrupted run).

## Reconstruction cache

`compute_distances()` writes every reconstruction to
`aeroblade_output/reconstructions/`, and joblib memoizes the computed
distances into `aeroblade_output/cache/`; both are reused on later runs
against the same images and AEs, so a repeat run costs seconds. Upstream
declares that joblib cache as `Memory(location="cache")` — relative to
the *current working directory*, which is now the repo root — so
`model.py` redirects it (`_patch_cache_location()`, plus a scoped
`chdir` around the import, since joblib creates the directory eagerly at
import time). Without that a stray `cache/` appears next to `run.py`.

Both directories are git-ignored and grow with each new dataset.
`run.py --all`'s cleanup does **not** delete them (only the clone and
the venv); remove `aeroblade_output/` by hand when you need the disk back.
