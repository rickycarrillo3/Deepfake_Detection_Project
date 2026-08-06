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
2. Creates an isolated venv, `venv`, on Python 3.9 (matches upstream's
   pinned torch build).
3. Installs `torch==2.1.1` / `torchvision==0.16.1` from the cu118 wheel
   index, then `rine/requirements.txt` and this directory's
   `requirements.txt`.
4. numpy gets bumped to 2.x by one of the above installs; RINE's
   `src/utils` needs 1.26.4, so the script pins it back down last.

Weights ship inside the vendored repo under `rine/ckpt/` — no separate
`weights/` download step is needed (unlike models that pull checkpoints
from Drive/S3 at load time).

## Run

Run from the repo root, using this model's venv:

```bash
models/rine/venv/bin/python run.py --model rine --dataset <dataset_name>
```

e.g. `--dataset sd3_faces_shards` scores `data/sd3_faces_shards/shard_01..10/` and writes
`results/sd3_faces_shards_rine_DM_predictions.csv`. All results share one flat `results/`
directory; the `rine_DM` middle comes from this model's `result_stem()`, which encodes the
`ncls` checkpoint so the four checkpoints' runs don't overwrite each other.

## Notes / assumptions carried over from the source notebook

- `get_our_trained_model(ncls="ldm", ...)` selects the LDM-trained
  checkpoint (one of several ckpts RINE ships — 1-class/2-class/4-class/LDM).
  Swap `ncls` in `model.py` if a different checkpoint is wanted.
- Falls back to CPU automatically if no CUDA device is available
  (the notebook hardcoded `cuda:0`).
- `opencv-python` (pulled in transitively) wants `numpy>=2`, which is
  incompatible with the `numpy==1.26.4` pin RINE's `src/utils` actually
  needs at import time — this is a known, harmless pip resolver warning,
  not a runtime error (confirmed working in the source notebook run).
