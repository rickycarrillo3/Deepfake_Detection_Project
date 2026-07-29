# gradient_boosting/

A **trained** ensemble over the per-detector scores in `results/`: a gradient-boosted tree model
takes 3–5 detectors' fake-probabilities for an image as its features and learns how to combine them.

This is the counterpart to `ensemble.py` at the repo root, which holds the **untrained** rules
(`mean`, `logodds`, `hard_or`). Those apply a fixed formula and need no labels; this one is fit on
data, which changes what it can express and what it requires.

This folder is **self-contained**. It shares no code with `neural_network/` — the two strategies can
be changed, re-run, or deleted independently. It uses only the project's existing foundation:
`loaders/` for the label rule and `metrics.py` for the threshold, the same two modules `run.py` and
`ensemble.py` already depend on.

```
gradient_boosting/
  data.py          pooled labeled design matrix, transforms, splits, class weights
  model.py         GradientBoostingEnsemble
  evaluation.py    metrics + artifact I/O
  train.py         CLI runner
  smoke_test.py    end-to-end check on synthetic scores
  requirements.txt
  setup.sh
```

## Quick start

```bash
# what can I select right now?
python3 gradient_boosting/train.py --list

# verify the setup without fitting (needs no scikit-learn)
python3 gradient_boosting/train.py \
    --models freqnet_GAN,rine1_DM,grag2021_latent_DM \
    --dataset ffhq_test --dataset flux_test --dry-run

# fit
python3 gradient_boosting/train.py \
    --models freqnet_GAN,rine1_DM,grag2021_latent_DM,cnndetection \
    --dataset ffhq_test --dataset flux_test --split by_shard --run-name gb_4model
```

Training is never implicit — it happens only when `--dry-run` is absent. `--dry-run` builds the real
design matrix, reports class balance and the split, prints per-detector baselines and the exact
booster configuration, then stops.

## What it learns that the untrained rules can't

`mean` and `logodds` apply one fixed formula to every image; `hard_or` fires if any single model
does. A booster fits shallow trees in stages, so it can express conditional structure instead:
*trust freqnet only when grag2021 also fires*, *rine's score only matters above 0.9*, *when these two
disagree, side with the third*. Those are exactly the interactions you'd hope to find given the
low-correlation model selection `failure_correlation.py` exists to do.

Trees also suit this table structurally: the features are few, on non-comparable scales, and each is
roughly monotone in the label. A tree splits on thresholds, so it's invariant to any monotone
rescaling of a column — the per-model calibration problem that forces `ensemble.py` to offer
`--norm` doesn't arise here.

## Why it pools datasets

Ground truth comes from the dataset name (`loaders.label_for_dataset`: `ffhq` → 0, anything else →
1), so **every dataset in this project is single-class**. `ensemble.py` evaluates one dataset at a
time because its rules need no labels. Gradient boosting is supervised: fit on a single dataset it
sees one label and learns to answer with the constant.

So a run must pool **at least one `ffhq*` dataset** (real) with **at least one generator dataset**
(fake). `data.py` enforces this rather than producing a meaningless 100%-accurate model.

Pooling also introduces a failure mode `ensemble.py` can't have. If ffhq was scored at 256×256 and
the generators at native resolution, the resize regime is perfectly confounded with the label and the
booster will happily learn your preprocessing. That's a hard error here, on top of the
within-dataset check `ensemble.py` already does.

Rows are keyed on `(dataset, filepath)`, never `filepath` alone — `run.py` writes paths
dataset-relative (`shard_01/00042.png`), so the same string exists in every dataset.

## Splits

The split decides which question the validation number answers. It matters more than any
hyperparameter here.

| `--split` | holds out | answers |
| --- | --- | --- |
| `random` (default) | stratified rows | does a booster help at all? optimistic — shard-mates land on both sides |
| `by_shard` | whole `shard_XX` dirs, stratified per dataset | does it generalize past the exact images it saw? |
| `by_dataset` | whole datasets (`--holdout-dataset`, repeatable) | does it transfer to a **generator it has never seen**? |

`by_dataset` is the question the project is ultimately built to answer, with a catch: holding out a
single dataset leaves validation single-class (AUROC undefined, accuracy = that class's recall), and
with only one real source you can't hold out a real dataset without leaving training with no real
images at all. `by_shard` fills that gap — it stratifies per dataset, so every dataset appears on
both sides and the per-dataset breakdown stays meaningful.

## Options worth knowing

| flag | default | note |
| --- | --- | --- |
| `--models` | required | 3–5 stems; order is part of the fitted model |
| `--backend` | `sklearn_gb` | `sklearn_hist` is much faster past ~1e5 rows |
| `--transform` | `raw` | `logit` also available; trees barely notice the difference |
| `--class-weight` | `balanced` | real images come from ffhq alone, so the pool is imbalanced by collection order, not by any real prior |
| `--n-estimators` | 300 | early stopping usually cuts this well short; the run prints how many rounds were actually used |
| `--learning-rate` | 0.05 | |
| `--max-depth` | 3 | with 3–5 features, deeper mostly memorizes |
| `--subsample` | 0.8 | `sklearn_gb` only |

`ensemble.py`'s `rank` normalization is deliberately unavailable: it's computed over whichever rows
you pass, so it leaks the validation distribution into the training features and can't be computed
for a single image at inference time.

## Environment

```bash
cd gradient_boosting && bash setup.sh
```

Builds an isolated CPU-only venv (`numpy`, `pandas`, `scikit-learn`, `joblib` — no torch). Isolated
for the same reason each model in `models/` has one: nothing here should force a version alignment
on any detector's pinned torch.

You can also skip it — `models/aide/` and `models/co-spy/` already have scikit-learn and pandas:

```bash
models/aide/venv/bin/python gradient_boosting/train.py ...
```

`--dry-run` needs only numpy + pandas, like the other analysis scripts.

## Artifacts

Written to `results/gradient_boosting/<run-name>/` (git-ignored, like the rest of `results/`):

```
config.json          stems (in order), transform, resize regime, datasets, split, hyperparams,
                     rounds actually used
metrics.json         train + validation metrics, with per-dataset breakdown
val_predictions.csv  dataset, filepath, score, label, Predicted label
model.joblib         the fitted booster
```

`config.json` records the **exact stem order** and `evaluation.check_feature_contract` enforces it.
Column order is part of the model: feeding the same three detectors in a different order produces
confident nonsense rather than an error.

## Testing

```bash
python3 gradient_boosting/smoke_test.py
```

Runs the whole path on synthetic scores in a temp directory — pooling at 3/4/5 models, every split
mode, every guard rail, both backends, metrics, and a save/load round-trip. Touches neither
`results/` nor `data/`. It's a wiring test, not a benchmark: the synthetic detectors are noisy by
construction, so the accuracies mean nothing beyond "finite and better than chance".
