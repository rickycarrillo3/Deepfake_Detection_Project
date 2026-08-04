# models/rf_ensemble/

A random-forest meta-model that fuses several already-set-up detectors' scores into one prediction.
You pick which 3-5 expert models to combine at training time; the forest learns how much to trust
each one.

Unlike every other model in `models/`, this one has **no vendored upstream repo and no pretrained
weights to download** — it's trained from scratch on this project's own output. It also doesn't
run any of the expert models' own code: it only reads their already-computed
`results/<dataset>_<model>_predictions.csv` files. That means it only works on images that every
chosen expert has *already* been scored on via the top-level `run.py` — it's a fusion-over-
existing-results model, not a standalone detector for a brand-new, never-scored image.

## Files

```
models/rf_ensemble/
├── model.py              # load() / predict() — the contract run.py calls
├── dataset.py             # builds the (expert_scores, label) training table from results/*.csv
├── train.py               # trains + evaluates the RandomForestClassifier, saves a .joblib
├── setup.sh                # builds the venv (scikit-learn, pandas, joblib, numpy — no GPU needed)
└── rf_ensemble.joblib      # produced by train.py; not committed until you train one
```

## Setup

```bash
cd models/rf_ensemble
bash setup.sh
```

No checkpoint download step — there's nothing pretrained to fetch. `setup.sh` just builds the venv
and installs `scikit-learn`/`pandas`/`joblib`/`numpy` (same conda/miniforge pip-bootstrap guard used
by every other model's `setup.sh` in this repo, for the same reason: `python -m venv` can silently
produce a broken `venv/bin/pip` on HPC clusters).

## Training

Pick 3-5 expert models (matching whatever `result_stem()` named their CSVs — check `results/` for
the exact stems, e.g. `aide_DM`, `co-spy_GAN`, `cnndetection`) and which datasets to train on:

```bash
models/rf_ensemble/venv/bin/python train.py \
    --datasets ffhq_faces flux_faces stylegan-0.5_faces stylegan-0.7_faces \
    --stems aide_DM co-spy_GAN cnndetection grag2021_latent \
    --out models/rf_ensemble/rf_ensemble.joblib
```

Every dataset you pass must already have a `results/<dataset>_<stem>_predictions.csv` for every
stem you pass — if one's missing, `train.py` errors out immediately telling you which model/dataset
combination to run first, rather than silently training on a partial expert set.

`train.py` prints, after training:
- test accuracy and ROC AUC on a held-out split
- a full `classification_report` (precision/recall/F1 for real vs. fake)
- **per-expert feature importance** — this is the most useful part for deciding whether an expert
  is actually pulling its weight in the ensemble or just adding noise

Other flags: `--n-estimators` (default 300), `--max-depth` (default unlimited), `--min-samples-leaf`
(default 1), `--test-fraction` (default 0.2), `--seed` (default 0).

### Label convention

Labels aren't read from a CSV column (the prediction CSVs don't have one — see the top-level
`run.py`'s own docstring). Instead, `dataset.py` derives label from the **dataset name**:
`ffhq_faces` is entirely real (label 0), every other dataset is entirely fake (label 1) — matching
the convention used throughout this project (`failure_correlation.py`, `models/gated_ensemble/`).
If you ever introduce a dataset that mixes real and fake images, this won't represent it correctly;
update `REAL_DATASETS` in `dataset.py` and this assumption throughout if that changes.

### How training examples are built

`dataset.py`'s `build_training_table()` inner-joins the chosen experts' CSVs on `filepath`, per
dataset — an image missing from even one expert's CSV (e.g. it errored during that model's own run)
is dropped for that dataset rather than included with a missing feature. Each surviving row becomes
one training example: a vector of `len(stems)` expert scores, plus the dataset-level label.

## Running it like every other model

Once trained, it fits the exact same contract as everything else in `models/`:

```bash
models/rf_ensemble/venv/bin/python run.py --model rf_ensemble --dataset ffhq_faces
```

`predict(image_path)`:
1. Recovers `(dataset_name, filepath)` from the image's path (assumes the standard
   `data/<dataset_name>/<filepath>` layout every other model relies on).
2. Looks up each expert's precomputed score for that exact file from
   `results/<dataset_name>_<stem>_predictions.csv` (loaded and cached per dataset on first use).
3. Feeds the score vector to the trained forest's `predict_proba()` and returns the fake-probability.

If any chosen expert hasn't been run on the dataset you're scoring, this raises a clear
`FileNotFoundError`/`KeyError` telling you exactly which model/dataset/file is missing, rather than
guessing or silently producing a wrong score.

### Using a different trained forest

`checkpoint_path=` (or the `RF_ENSEMBLE_CHECKPOINT` env var) overrides which `.joblib` to load, e.g.
if you train more than one forest with different expert sets:

```bash
models/rf_ensemble/venv/bin/python run.py --model rf_ensemble --dataset ffhq_faces \
    --model-arg checkpoint_path=models/rf_ensemble/rf_ensemble_v2.joblib
```

`result_stem()` names the output CSV after the checkpoint file used (`rf_ensemble` for the default
name, `rf_ensemble_<checkpoint_name>` otherwise), so two differently-trained forests don't overwrite
each other's results.

## Register it in the top-level `run.py`

Add one line to `MODELS`:

```python
MODELS = {
    ...
    "rf_ensemble": "models/rf_ensemble",
}
```

No `VARIANT_ARGS` entry is needed unless you plan to train and keep multiple forests (different
expert sets) side by side and want `--all` to run all of them automatically.
