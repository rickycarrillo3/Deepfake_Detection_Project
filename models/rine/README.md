# models/rine/

Wraps [mever-team/rine](https://github.com/mever-team/rine) ("Leveraging Representations from
Intermediate Encoder-blocks for Synthetic Image Detection"), which scores an image from the
intermediate blocks of a CLIP encoder rather than its final representation.

## Setup

```bash
cd models/rine && bash setup.sh
```

**This detector needs Python 3.9** — upstream pins its torch build against it. On a cluster, load
a matching module and point `PYTHON` at it:

```bash
module load python/3.9 && PYTHON=python3.9 bash setup.sh
```

`setup.sh` is safe to re-run and skips completed steps. What it does, and why each step is there:

1. Clones upstream to `rine/` (contains `src/`, `ckpt/`, `demo/`, `scripts/`).
2. **Symlinks `ckpt` at the project root** to `models/rine/rine/ckpt`. Upstream's
   `get_our_trained_model()` loads the relative path `ckpt/model_<setting>_trainable.pth`, and
   `run.py` is invoked from the repo root, so without the symlink the checkpoint lookup misses.
3. Creates an isolated `venv/`.
4. Installs the vendored repository's `requirements.txt` and this directory's, which is where
   `torch==2.2.0` / `torchvision==0.17.0` are pinned. An explicit cu118 wheel-index install sits
   commented out in `setup.sh` — uncomment it if you need that build.
5. Pins `numpy`/`scipy`/`scikit-learn` back down **last**: one of the earlier installs bumps numpy
   to 2.x, and RINE's `src/utils` needs 1.26.4 at import time.

Weights ship inside the vendored repository under `rine/ckpt/`, so there is no separate download.

## Choosing a checkpoint — `ncls`

RINE ships four checkpoints, selected with `--model-arg ncls=<value>`. No code edit is needed:
`load(ncls=...)` passes it to upstream's `get_our_trained_model`, coercing to `int` for anything
but `"ldm"`, since `--model-arg` always arrives as a string.

| `ncls` | checkpoint | trained on | result stem |
|---|---|---|---|
| `ldm` (default) | `model_ldm_trainable.pth` | latent diffusion | `rine_DM` |
| `1` | `model_1class_trainable.pth` | ProGAN, 1 class | `rine1_GAN` |
| `2` | `model_2class_trainable.pth` | ProGAN, 2 classes | `rine2_GAN` |
| `4` | `model_4class_trainable.pth` | ProGAN, 4 classes | `rine4_GAN` |

The `ncls` sits in the model slot of the filename so the four don't overwrite each other's CSVs.
All four are listed in `VARIANT_ARGS` in the top-level `run.py`, so a sweep (`--all` /
`--models rine`) runs each of them.

## Running

```bash
models/rine/venv/bin/python run.py --model rine --dataset flux_test
models/rine/venv/bin/python run.py --model rine --dataset flux_test --model-arg ncls=4
# -> results/flux_test_rine_DM_predictions.csv
#    results/flux_test_rine4_GAN_predictions.csv
```

Schema in [`results/README.md`](../../results/README.md).

## Implementation notes

- The device is resolved at load time, falling back to CPU when no CUDA device is available.
- `opencv-python` (pulled in transitively) declares `numpy>=2`, which conflicts with the
  `numpy==1.26.4` that RINE's `src/utils` needs. This surfaces as a pip resolver warning rather
  than a runtime error — step 5 above is what leaves the environment working.
