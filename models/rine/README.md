# models/rine/

Wraps [mever-team/rine](https://github.com/mever-team/rine) ("Leveraging
Representations from Intermediate Encoder-blocks for Synthetic Image
Detection") behind the repo's `load()` / `predict(image_path) -> score`
contract, so the shared top-level `run.py` can be pointed at any dataset under
`data/` via `loaders.iter_dataset`.

Ported from a one-off Colab notebook that cloned the repo, ran
`demo/demo.py` as a smoke test, then scored 10 shards with a standalone
`test.py` — the top-level `run.py` is the equivalent, repo-integrated version.

## Setup

This model gets its own isolated environment (its dependencies conflict
with other detectors' torch/numpy pins), following the co-spy template.

```bash
cd models/rine
bash setup.sh
```

`setup.sh` does the following (safe to re-run — skips steps already done):

1. Vendors the upstream repo (contains `src/`, `ckpt/`, `demo/`, `scripts/`)
   via `git clone https://github.com/mever-team/rine.git rine`.
2. Symlinks `ckpt` at the **project root** to `models/rine/rine/ckpt`.
   Upstream's `get_our_trained_model()` loads the relative path
   `ckpt/model_<setting>_trainable.pth`, and `run.py` is invoked from the
   repo root, so without the symlink the checkpoint lookup misses.
3. Creates an isolated `venv/`. The interpreter is overridable per the repo
   convention (`PYTHON="${PYTHON:-python3}"`) — upstream pins its torch build
   against Python 3.9, so on a cluster load a matching module and point
   `PYTHON` at it: `module load python/3.9 && PYTHON=python3.9 ./setup.sh`.
4. Installs `rine/requirements.txt` (the vendored repo's own deps) and this
   directory's `requirements.txt`, which is where `torch==2.2.0` /
   `torchvision==0.17.0` are pinned. The explicit cu118 wheel-index install is
   commented out in `setup.sh` — uncomment it if you need that build.
5. numpy gets bumped to 2.x by one of the above installs; RINE's `src/utils`
   needs 1.26.4, so the script pins numpy/scipy/scikit-learn back down last.

Weights ship inside the vendored repo under `rine/ckpt/` — no separate
`weights/` download step is needed (unlike models that pull checkpoints
from Drive/S3 at load time).

## Run

Run from the repo root, using this model's venv:

```bash
models/rine/venv/bin/python run.py --model rine --dataset <dataset_name>
```

e.g. `--dataset sd3_faces_shards` scores `data/sd3_faces_shards/shard_01../` and writes
`results/sd3_faces_shards_rine_DM_predictions.csv` (schema in
[`results/README.md`](../../results/README.md)).

## Choosing a checkpoint — `ncls`

RINE ships four checkpoints in `rine/ckpt/`, selected with `--model-arg ncls=<value>` (no code
edit — `load(ncls=...)` passes it to upstream's `get_our_trained_model`, coercing to `int` for
anything but `"ldm"`, since `--model-arg` always arrives as a string):

| `ncls` | checkpoint | trained on | result stem |
|---|---|---|---|
| `ldm` (default) | `model_ldm_trainable.pth` | latent diffusion | `rine_DM` |
| `1` | `model_1class_trainable.pth` | ProGAN, 1 class | `rine1_GAN` |
| `2` | `model_2class_trainable.pth` | ProGAN, 2 classes | `rine2_GAN` |
| `4` | `model_4class_trainable.pth` | ProGAN, 4 classes | `rine4_GAN` |

The `ncls` sits in the model slot of the filename so the four don't overwrite each other's CSVs.
All four are listed in `VARIANT_ARGS` in the top-level `run.py`, so a sweep (`--all` /
`--models rine`) runs each of them:

```bash
models/rine/venv/bin/python run.py --model rine --dataset flux_faces --model-arg ncls=4
# -> results/flux_faces_rine4_GAN_predictions.csv
```

## Notes / assumptions carried over from the source notebook
- Falls back to CPU automatically if no CUDA device is available
  (the notebook hardcoded `cuda:0`).
- `opencv-python` (pulled in transitively) wants `numpy>=2`, which is
  incompatible with the `numpy==1.26.4` pin RINE's `src/utils` actually
  needs at import time — this is a known, harmless pip resolver warning,
  not a runtime error (confirmed working in the source notebook run).
