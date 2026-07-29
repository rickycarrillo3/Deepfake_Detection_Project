# neural_network/

A **trained** ensemble over the per-detector scores in `results/`: a small MLP takes 3–5 detectors'
fake-probabilities for an image as its input nodes and outputs a single sigmoid value, which is the
classifier.

```
image ──> detector 1 ─┐
      ──> detector 2 ─┤
      ──> detector 3 ─┼──> [ MLP ] ──> sigmoid ──> P(fake) ──> >= 0.5 ? fake
      ──> detector 4 ─┤
      ──> detector 5 ─┘
    (run.py already did this,          this folder
     the scores are the results CSVs)
```

The image never reaches this folder. `run.py` has already scored it with each detector and written
the score to `results/<dataset>_<stem>_predictions.csv`, so the input layer is 3–5 numbers in a fixed
column order. That's what makes this trainable on a laptop: the expensive part is cached on disk and
this is a few hundred parameters on top. The input width comes from how many models you select —
three stems build three input nodes, five build five.

This is the counterpart to `ensemble.py` at the repo root, which holds the **untrained** rules
(`mean`, `logodds`, `hard_or`). Those apply a fixed formula and need no labels; this one is trained.

This folder is **self-contained**. It shares no code with `gradient_boosting/` — the two strategies
can be changed, re-run, or deleted independently. It uses only the project's existing foundation:
`loaders/` for the label rule and `metrics.py` for the threshold, the same two modules `run.py` and
`ensemble.py` already depend on.

```
neural_network/
  data.py          pooled labeled design matrix, transforms, splits, pos_weight
  model.py         MLPEnsemble
  evaluation.py    metrics (incl. dependency-free AUROC) + artifact I/O
  train.py         CLI runner
  smoke_test.py    end-to-end check on synthetic scores
  requirements.txt
  setup.sh
```

## Quick start

```bash
# what can I select right now?
python3 neural_network/train.py --list

# verify the setup without training (needs no torch)
python3 neural_network/train.py \
    --models freqnet_GAN,rine1_DM,grag2021_latent_DM \
    --dataset ffhq_test --dataset flux_test --dry-run

# train
python3 neural_network/train.py \
    --models freqnet_GAN,rine1_DM,grag2021_latent_DM,cnndetection \
    --dataset ffhq_test --dataset flux_test --split by_shard --run-name nn_4model
```

Training is never implicit — it happens only when `--dry-run` is absent. `--dry-run` builds the real
design matrix, reports class balance and the split, prints per-detector baselines and the exact
architecture (layer sizes, parameter count, loss, optimizer), then stops.

## The output head

One linear unit to a single logit, trained with `BCEWithLogitsLoss`. **That is the sigmoid** —
`BCEWithLogits` applies it internally in a log-sum-exp-stable way rather than doing sigmoid-then-log
in two steps, which overflows once a confident logit passes roughly ±30. So there's deliberately no
`nn.Sigmoid` in the `Sequential`; `predict_proba()` applies `torch.sigmoid` explicitly at inference,
and what comes out is a probability in (0, 1) thresholded at `metrics.THRESHOLD` exactly like every
other score in this project.

## Input scaling

Unlike a tree, a network is very much not invariant to input scale, so this is the part most worth
understanding.

Several detectors saturate hard — freqnet emits an exact `0.0` — so in probability space most of the
mass sits on two spikes at the ends of [0, 1] and the first linear layer has almost nothing to
separate. That's why `--transform logit` is the **default here** (the gradient boosting folder
defaults to `raw`, since trees don't care). The logit spreads those spikes back out and is the scale
the base detectors' own final layers were linear in to begin with.

On top of that, `model.py` z-scores each column using **training-set statistics only**, frozen into
the checkpoint. Validation and later inference are transformed by those same numbers — not their own,
which would leak across the split and would be uncomputable for a single image at inference time.

## Why it pools datasets

Ground truth comes from the dataset name (`loaders.label_for_dataset`: `ffhq` → 0, anything else →
1), so **every dataset in this project is single-class**. `ensemble.py` evaluates one dataset at a
time because its rules need no labels. A network trained with BCE is supervised: fit on a single
dataset it sees one label, drives the output bias to one extreme, and reaches zero loss having
learned nothing.

So a run must pool **at least one `ffhq*` dataset** (real) with **at least one generator dataset**
(fake). `data.py` enforces this.

Pooling also introduces a failure mode `ensemble.py` can't have. If ffhq was scored at 256×256 and
the generators at native resolution, the resize regime is perfectly confounded with the label and the
network will learn your preprocessing. That's a hard error here, on top of the within-dataset check
`ensemble.py` already does.

Rows are keyed on `(dataset, filepath)`, never `filepath` alone — `run.py` writes paths
dataset-relative (`shard_01/00042.png`), so the same string exists in every dataset.

## Splits

The split decides which question the validation number answers, and here it does double duty: it's
both the reported score **and** the early-stopping signal.

| `--split` | holds out | answers |
| --- | --- | --- |
| `random` (default) | stratified rows | does a network help at all? optimistic — shard-mates land on both sides |
| `by_shard` | whole `shard_XX` dirs, stratified per dataset | does it generalize past the exact images it saw? |
| `by_dataset` | whole datasets (`--holdout-dataset`, repeatable) | does it transfer to a **generator it has never seen**? |

`by_dataset` is the question the project is ultimately built to answer, with a catch: holding out a
single dataset leaves validation single-class (AUROC undefined, accuracy = that class's recall, and
early stopping selecting on a one-sided signal), and with only one real source you can't hold out a
real dataset without leaving training with no real images at all. `by_shard` fills that gap — it
stratifies per dataset, so every dataset appears on both sides.

## Options worth knowing

| flag | default | note |
| --- | --- | --- |
| `--models` | required | 3–5 stems, one input node each; order is part of the trained model |
| `--hidden` | `16,8` | deliberately tiny; `--hidden ''` gives logistic regression on the scores |
| `--transform` | `logit` | see input scaling above |
| `--class-weight` | `balanced` | sets the loss's `pos_weight`; real images come from ffhq alone, so the pool is imbalanced by collection order, not by any real prior |
| `--epochs` | 200 | with `--patience 20`; best weights by validation loss are restored, not the last epoch's |
| `--learning-rate` | 1e-3 | Adam |
| `--weight-decay` | 1e-4 | |
| `--dropout` | 0.0 | on 3–5 inputs, dropout usually hurts |
| `--device` | `auto` | CPU is fine — this is a few hundred parameters |

Why the default is so small: with 3–5 inputs there's little to learn beyond a smooth non-linear
surface, and this is a meta-learner sitting on a handful of single-class shards — a wide net
memorizes which dataset a score pattern came from. Start at `16,8`, look at
`training_history.json`, and only scale up if the loss curves say to. `--hidden ''` is a genuinely
useful baseline: if the MLP can't beat plain logistic regression on the same scores, the extra
capacity isn't buying anything.

`ensemble.py`'s `rank` normalization is deliberately unavailable: it's computed over whichever rows
you pass, so it leaks the validation distribution into the training features and can't be computed
for a single image at inference time.

## Environment

```bash
cd neural_network && bash setup.sh
```

Builds an isolated CPU-only venv (`numpy`, `pandas`, `torch` — no torchvision, no scikit-learn).
Isolated for the same reason each model in `models/` has one: nothing here should force a version
alignment on any detector's pinned torch. AUROC is computed from ranks in `evaluation.py` rather
than imported from scikit-learn, so torch is the only real dependency.

You can also skip it — every model venv already has torch and pandas:

```bash
models/freqnet/venv/bin/python neural_network/train.py ...
```

`--dry-run` needs only numpy + pandas, like the other analysis scripts.

## Artifacts

Written to `results/neural_network/<run-name>/` (git-ignored, like the rest of `results/`):

```
config.json             stems (in order), transform, resize regime, datasets, split,
                        hyperparams, best epoch, epochs run
metrics.json            train + validation metrics, with per-dataset breakdown
val_predictions.csv     dataset, filepath, score, label, Predicted label
model.pt                weights + the frozen input scaler
training_history.json   per-epoch train/val loss — the first thing to look at
```

`config.json` records the **exact stem order** and `evaluation.check_feature_contract` enforces it.
Column order is part of the model — the frozen per-column scaler is aligned to it — so feeding the
same three detectors in a different order produces confident nonsense rather than an error.

## Testing

```bash
python3 neural_network/smoke_test.py
```

Runs the whole path on synthetic scores in a temp directory — pooling at 3/4/5 models, every split
mode, every guard rail, architecture and parameter counts, AUROC, the training loop, and a save/load
round-trip. Touches neither `results/` nor `data/`.

Without torch installed it still runs and covers everything except the training loop, reporting that
step as skipped. To cover it, re-run under a venv that has torch — `neural_network/setup.sh`, or any
model venv:

```bash
models/freqnet/venv/bin/python neural_network/smoke_test.py
```

It's a wiring test, not a benchmark: the synthetic detectors are noisy by construction, so the
accuracies mean nothing beyond "finite and better than chance".
