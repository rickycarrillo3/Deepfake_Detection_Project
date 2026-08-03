# models/

Holds one subdirectory per pretrained deepfake detector under test.

## Layout

```
models/<model_name>/
├── model.py              # load() / predict() — the contract run.py calls
├── setup.sh              # clones upstream, builds the venv, fetches weights
├── requirements.txt      # or environment.yml — this model's own dependencies
├── venv/ (or .venv/)     # isolated environment, git-ignored
├── weights/              # model checkpoint(s), git-ignored
└── ...                   # vendored/cloned copy of the model's GitHub repo, git-ignored
```

There is **no per-model `run.py`** — the top-level `run.py` runs every model. See
"Adding a model" below.

Note the mapping is not strictly one directory per detector:
`models/DMimageDetection/` holds two detectors (`dmimagedetection_DM`, `dmimagedetection_GAN`) in
subdirectories, sharing one venv and one weights download.

## Per-model isolation

Each model directory manages its **own environment**. Pretrained detectors pulled from different
GitHub repos commonly pin conflicting torch/tf versions, so there is intentionally no shared
top-level `requirements.txt`. The shared `run.py` needs only `pandas` + `tqdm`, which every model
venv already has, so it runs under each model's own interpreter:

```bash
models/<model_name>/venv/bin/python run.py --model <model_name> --dataset <dataset_name>
```

## Wrapper interface (required)

Each `model.py` must expose, at **module level** — a class alone is not enough, since this is what
`run.py` calls:

- `load(**kwargs)` — load weights and prepare the model for inference
- `predict(image_path) -> score` — return a float score in `[0, 1]` for a single image

Implementing the detector as a class is fine; just add a thin module-level shim delegating to it.
See the bottom of `models/freqnet/model.py` for the pattern:

```python
_DEFAULT_MODEL = None

def load(**kwargs):
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = FreqNetModel(**kwargs)
    _DEFAULT_MODEL.load()
    return _DEFAULT_MODEL

def predict(image_path):
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)
```

`model.py` must resolve its weights and vendored code relative to `__file__`, never the current
working directory — `run.py` imports it from the repo root.

Anything model-specific that `load()` needs (a checkpoint path, a weight variant) is passed through
from the command line with `--model-arg key=value`, so the runner never needs to know what it means.

## Adding a model

1. Write `models/<model_name>/model.py` exposing `load()` / `predict()` as above.
2. Write `models/<model_name>/setup.sh` (clone upstream, build the venv, fetch weights). Build the
   venv with an overridable interpreter — `PYTHON="${PYTHON:-python3}"` — so a cluster's
   module-loaded Python can be used; do not hardcode one.
3. Add one line to the `MODELS` registry in the top-level `run.py`.

That's it — do not write a `run.py`. The per-model runners used to be copy-pasted near-duplicates
that drifted apart and had to be bug-fixed one by one; they were consolidated deliberately.

## Exception: aeroblade

`models/aeroblade/` keeps its own `run.py`. It scores a whole shard per call via
`predict_batch(shard_dir)` and emits extra per-autoencoder distance columns, so it does not fit the
one-image-at-a-time contract. It is the only exception.
