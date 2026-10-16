# Adding a detector

Four steps. The whole point of the shared runner is that this is all there is — there is **no
per-model `run.py`** to copy, and nothing in `run.py` needs to learn what your detector does.

```
models/<model_name>/
├── model.py          # 1. the contract run.py calls
├── setup.sh          # 2. clone upstream, build the venv, fetch weights
└── requirements.txt  #    this detector's own dependencies
                      # 3. add the clone path to the root .gitignore
                      # 4. add one line to MODELS in run.py
```

`models/freqnet/` is the smallest complete example; read it alongside this document.

## 1. Write `model.py`

It must expose two names **at module level** — a class alone is not enough, since module-level
`load` and `predict` are literally what `run.py` looks up:

- `load(**kwargs)` — load weights, prepare for inference
- `predict(image_path) -> float` — a score in `[0, 1]`, the predicted fake-probability

Writing the detector as a class is the normal approach; add a thin module-level shim delegating to
it. The skeleton, following `models/freqnet/model.py`:

```python
import os
import sys

import torch

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "<UpstreamClone>")
CHECKPOINT_PATH = os.path.join(REPO_DIR, "<weights>.pth")


class MyDetector:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load(self):
        # Vendored upstream code goes on sys.path here, not at import time.
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
        from networks.mydetector import MyNet

        self.model = MyNet()
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=True)
        self.model.to(self.device).eval()
        return self

    def predict(self, image_path):
        ...
        return float(prob)


_DEFAULT_MODEL = None


def load(**kwargs):
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = MyDetector(**kwargs)
    _DEFAULT_MODEL.load()
    return _DEFAULT_MODEL


def predict(image_path):
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)


def result_stem():
    """The '<model>_<family>' middle of the results filename."""
    return "mydetector_GAN"
```

Three requirements that are easy to miss:

- **Resolve every path relative to `__file__`, never the CWD.** `run.py` imports your `model.py`
  from the repo root, so a relative path to weights or vendored code will not resolve.
- **Fall back to CPU** when `torch.cuda.is_available()` is false, so the detector is runnable
  without a GPU.
- **Load state dicts with `strict=True`.** A silently partial load produces plausible-looking
  scores rather than an error.

### `result_stem()`

Optional but expected — every registered detector defines one. It sets the `<model>_<family>`
middle of the output filename, which is what keeps runs from colliding in the flat `results/`
directory:

```
results/<dataset_name>_<result_stem()>_predictions.csv
```

Return the generation family the detector was trained on (`"mydetector_GAN"`, `"mydetector_DM"`).
If a `--model-arg` meaningfully changes the weights, fold it in so the variants land in separate
CSVs — `co-spy` keys on `train_dataset`, `aide` on `checkpoint`, `rine` on `ncls`. Omit it and the
runner falls back to the bare registry name, with no family.

### Optional schema hooks

Read by `run.py` with `getattr` defaults, so you only define what you need:

| name | default | effect |
|---|---|---|
| `SCORE_COLUMN` | `"score"` | renames the score column |
| `SCORE_PRECISION` | `4` | decimals the score and extra columns are rounded to |
| `PREDICTED_LABEL` | `True` | `False` drops `Predicted label` and the console accuracy |
| `SUPPORTS_RESIZE` | `True` | `False` makes `--resize` an error for this detector |
| `extra_columns(image_path)` | — | dict of extra columns, written right after the score |

These exist for scores that are genuinely **not** fake-probabilities, not as a general escape
hatch. Only `aeroblade` uses them, because it is training-free and returns a raw reconstruction
distance for which no real/fake cutoff is defined. A detector returning a calibrated
fake-probability should use the default schema, so its column is comparable to the others.

## 2. Write `setup.sh`

Clone upstream, build an isolated venv, fetch weights. Keep environments per-directory — upstream
repos pin conflicting torch/TF versions, so there is deliberately no shared top-level requirements
file.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "<UpstreamClone>" ]; then
  git clone https://github.com/<owner>/<repo>.git <UpstreamClone>
fi

# Interpreter used to build the venv. Override on clusters where the system
# python3 is old, e.g. after `module load python/3.12.12`:
#   PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

"$PYTHON" -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
```

Use `PYTHON="${PYTHON:-python3}"` rather than hardcoding an interpreter — clusters need to select
a module-loaded Python. If upstream's pins constrain the version, check it and fail with a message
naming the range (see `baselines/P2G/setup.sh`) instead of letting pip produce a confusing
resolution error.

## 3. Add the clone to the root `.gitignore`

Add your `git clone` target to the **vendored-clone block in the root `.gitignore`** — the one
list, not a per-model file. Keeping them in one block is what makes coverage checkable by reading
a single list.

`git status` after running your `setup.sh` is the test that you did it: a clean tree means the
clone, the venv and the weights are all ignored.

## 4. Register it

One line in the `MODELS` registry in the top-level `run.py`:

```python
MODELS = {
    ...
    "mydetector": "models/mydetector",
}
```

If the detector ships several meaningfully different checkpoints, add them to `VARIANT_ARGS` in
the same file so sweeps expand it automatically:

```python
VARIANT_ARGS = {
    "mydetector": [{"checkpoint": "progan"}, {"checkpoint": "sd14"}],
}
```

Two registry details apply only to unusual layouts: `VENV_OVERRIDE` /`SHARED_CLONE_ROOTS` /
`SHARED_VENV_ROOTS` for detectors that share one clone and venv (as the two `DMimageDetection`
entries do), and `NO_CLONE_MODELS` for one whose architecture is transcribed into this repo rather
than cloned, so sweeps do not read a deliberately absent clone as a broken setup.

## 5. Verify

```bash
python3 run.py --list                                               # it appears in the registry
models/mydetector/venv/bin/python run.py --model mydetector \
    --dataset ffhq_test --limit 10                                  # it scores
python3 run.py --models mydetector --dataset ffhq_test --dry-run    # sweeps find its venv and clone
```

Check that `results/ffhq_test_mydetector_<family>_predictions.csv` appeared with ten rows and
scores inside `[0, 1]`, and that the dry run reports a venv and a clone rather than "nothing to
remove" — the latter means a sweep would re-run your `setup.sh` on every pass.

A sanity check worth doing once: run the same image through upstream's own inference script and
confirm the score matches. A wrong preprocessing step or an inverted output convention produces
entirely reasonable-looking numbers.
