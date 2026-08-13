# models/

Holds one subdirectory per pretrained deepfake detector under test, using the **original authors'**
checkpoints. (The six detectors retrained by the TrueFake paper's authors live in
[`baselines/`](../baselines/README.md) instead, `tf-` prefixed — same contract, different
provenance.)

## The nine detectors

| registry name | dir | upstream |
|---|---|---|
| `freqnet` | `freqnet/` | chuangchuangtan/FreqNet-DeepfakeDetection — frequency-domain CNN |
| `grag2021` | `grag2021/` | grip-unina/GANimageDetection (Gragnaniello et al., ICME 2021) |
| `cnndetection` | `cnndetection/` | PeterWang512/CNNDetection (Wang et al., CVPR 2020) |
| `co-spy` | `co-spy/` | Megum1/CO-SPY — semantic + artifact + fusion |
| `aide` | `aide/` | shilinyan99/AIDE |
| `rine` | `rine/` | mever-team/rine — intermediate CLIP encoder blocks |
| `DMimageDetection_DM` | `DMimageDetection/DMimageDetection_DM/` | grip-unina/DMimageDetection (Corvi et al., ICASSP 2023), latent-diffusion checkpoint |
| `DMimageDetection_GAN` | `DMimageDetection/DMimageDetection_GAN/` | same repo, ProGAN checkpoint |
| `aeroblade` | `aeroblade/` | jonasricker/aeroblade (CVPR 2024) — training-free reconstruction error |

`grag2021` and `DMimageDetection` are easy to conflate: same lab and overlapping co-authors, but
different repos, papers and weights. The upstream checkpoint keys `Grag2021_progan` /
`Grag2021_latent` stay in each `DMimageDetection` `model.py`'s `MODEL_NAME` — do not rename them.

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
`models/DMimageDetection/` holds two detectors (`DMimageDetection_DM`, `DMimageDetection_GAN`) in
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

## Naming the output — `result_stem()`

A `model.py` may also define a module-level `result_stem()`, called with no arguments **after**
`load()`. It returns the `<model>_<family>` middle of the output filename
(`results/<dataset>_<stem>_predictions.csv`), which is how the generation family a detector was
trained on reaches the CSV name — and how a model whose `--model-arg` meaningfully changes its
weights keeps those runs in separate files now that all results share one flat directory:

```python
def result_stem():
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before result_stem().")
    family = "GAN" if _DEFAULT_MODEL.train_dataset == "progan" else "DM"
    return f"co-spy_{family}"
```

Every model registered today defines one; a model that doesn't falls back to a bare
`<model_name>`. Because it runs after `load()`, it can read whatever `load()` resolved. See
`results/README.md` for the full stem-per-model table.

## Models with several checkpoints

When one model ships more than one meaningfully-different weight set, list those `--model-arg`
combinations in `VARIANT_ARGS` in the top-level `run.py` so a sweep (`--all` / `--baselines` /
`--models`) runs each of them rather than only the default. Registered today:

| model | variants |
|---|---|
| `co-spy` | `train_dataset=progan`, `sd-v1_4` |
| `aide` | `checkpoint=genimage`, `progan`, `sd14` |
| `rine` | `ncls=ldm`, `1`, `2`, `4` |

Their `result_stem()`s must differ across those variants, or the runs overwrite each other's CSVs.
A model absent from `VARIANT_ARGS` simply runs once with no extra args.

## Adding a model

1. Write `models/<model_name>/model.py` exposing `load()` / `predict()` as above.
2. Write `models/<model_name>/setup.sh` (clone upstream, build the venv, fetch weights). Build the
   venv with an overridable interpreter — `PYTHON="${PYTHON:-python3}"` — so a cluster's
   module-loaded Python can be used; do not hardcode one.
3. Add one line to the `MODELS` registry in the top-level `run.py` (plus a `VARIANT_ARGS` entry if
   the detector ships several checkpoints).

Check it end to end with a short run before turning it loose on a full dataset:

```bash
models/<model_name>/venv/bin/python run.py --model <model_name> --dataset ffhq_test --limit 10
```

That's it — do not write a `run.py`. The per-model runners used to be copy-pasted near-duplicates
that drifted apart and had to be bug-fixed one by one; they were consolidated deliberately.

## Optional CSV-schema hooks

A `model.py` whose output does not fit the default CSV schema may define these module-level
names; `run.py` reads them with `getattr` defaults, so nothing else needs to change:

| hook | default | effect |
|---|---|---|
| `SCORE_COLUMN` | `"score"` | renames the score column |
| `SCORE_PRECISION` | `4` | decimals the score and extra columns are rounded to |
| `PREDICTED_LABEL` | `True` | `False` drops the `Predicted label` column and the console accuracy line |
| `SUPPORTS_RESIZE` | `True` | `False` makes `--resize` an error for this model |
| `extra_columns(image_path)` | — | dict of extra columns, written right after the score |

Only `models/aeroblade/` uses them today: it is training-free and returns a raw LPIPS
reconstruction distance rather than a fake-probability, so there is no `[0, 1]` score to
threshold. It also scores a whole directory per upstream call, and `predict()` caches by parent
directory to keep that batching under the one-image-at-a-time contract — which is why it declines
`--resize`. See `models/aeroblade/README.md`. Reach for these hooks only when a model genuinely
does not produce a fake-probability; they are not a general escape hatch.
