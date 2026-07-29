"""Design matrix for the gradient-boosting ensemble.

Self-contained on purpose: this folder shares no code with neural_network/, so
the two strategies can be changed, re-run, or thrown away independently. It uses
only the project's existing foundation — loaders/ for the label rule and
metrics.py for the threshold — the same two modules run.py and ensemble.py
already depend on.

What it builds
--------------
One row per image, one column per selected detector, holding that detector's
fake-probability from results/<dataset>_<stem>_predictions.csv, plus the image's
ground-truth label. That table is what the booster is fit on: 3-5 feature
columns and a binary target.

Why it has to pool datasets
---------------------------
Ground truth comes from the dataset name (loaders.label_for_dataset: a name
containing 'ffhq' -> 0/real, anything else -> 1/fake), so every dataset in this
project is single-class. ensemble.py can evaluate one dataset at a time because
mean/logodds/hard_or need no labels. Gradient boosting is supervised: fit on a
single dataset it sees one label and learns to answer with the constant.

So a run must pool at least one ffhq* dataset (the real class) with at least one
generator dataset (the fake class). build_pooled() enforces that rather than
letting it produce a meaningless 100%-accurate model.

Row identity
------------
run.py writes filepath dataset-relative ('shard_01/00042.png'), so the same
string appears in every dataset's CSVs. Rows are keyed on (dataset, filepath),
never filepath alone, or ffhq's shard_01/00042.png collides with flux's.

Resize regimes
--------------
run.py records the --resize regime of each run in a `resize` column so runs are
not silently mixed. Two checks here, both fatal:

  within a dataset   mixing models scored at different regimes (what
                     ensemble.py already guards).
  across datasets    if ffhq was scored at 256x256 and the generators at native
                     resolution, the regime is perfectly confounded with the
                     label and the booster will happily learn the preprocessing
                     instead of the forgery. Pooling is what makes this possible,
                     so the check belongs here.

Feature transform
-----------------
--transform raw (default) or logit. Trees split on thresholds, so they are
invariant to any monotone rescaling of a column — that is exactly the per-model
calibration problem that forces ensemble.py to offer --norm at all, and it is
why the booster barely notices the difference. raw is the default here because
the numbers stay readable as probabilities. logit is offered for comparison
against the neural network, which does care.

ensemble.py's `rank` normalization is deliberately not available: it is computed
over whichever rows you pass, so it leaks the validation rows' distribution into
the training features and cannot be computed for a single image at inference.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FOLDER)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loaders.dataset import label_for_dataset  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")
_SUFFIX = "_predictions.csv"

# Clamp probabilities off {0, 1} before logit(): freqnet emits an exact 0.0,
# which would otherwise map to -inf. Same value ensemble.py uses.
_EPS = 1e-6

TRANSFORMS = ("raw", "logit")
SPLITS = ("random", "by_shard", "by_dataset")
CLASS_WEIGHTS = ("balanced", "none")

# The booster takes 3-5 detector scores as its features. Below 3 a trained
# combiner is not worth the machinery over a plain average; above 5, with only a
# handful of single-class shards behind each column, it starts fitting the
# dataset roster rather than the detectors. A project decision, not a hard limit.
MIN_MODELS = 3
MAX_MODELS = 5


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_datasets():
    """Dataset names under data/ that have at least one prediction CSV."""
    datasets = []
    if not os.path.isdir(DATA_DIR):
        return datasets
    for name in sorted(os.listdir(DATA_DIR)):
        if os.path.isdir(os.path.join(DATA_DIR, name)) and model_stems_for(name):
            datasets.append(name)
    return datasets


def model_stems_for(dataset_name):
    """Model stems ('<model>_<family>') that have a CSV for this dataset.

    Anchored on the known dataset name rather than split on '_', since both
    dataset and model names contain underscores (ffhq_test, grag2021_latent_DM,
    co-spy_sd-v1_4_DM, ...).
    """
    prefix = f"{dataset_name}_"
    stems = []
    if not os.path.isdir(RESULTS_DIR):
        return stems
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith(prefix) and fname.endswith(_SUFFIX):
            stem = fname[len(prefix):-len(_SUFFIX)]
            if stem:
                stems.append(stem)
    return stems


def csv_path(dataset_name, stem):
    return os.path.join(RESULTS_DIR, f"{dataset_name}_{stem}{_SUFFIX}")


def common_stems(datasets):
    """Stems with a CSV for *every* dataset in `datasets`.

    A pooled matrix needs one column per model spanning all rows, so a model
    missing from one dataset cannot contribute a column.
    """
    if not datasets:
        return []
    return sorted(set.intersection(*[set(model_stems_for(d)) for d in datasets]))


def validate_selection(stems, min_models=MIN_MODELS, max_models=MAX_MODELS):
    if not (min_models <= len(stems) <= max_models):
        raise SystemExit(
            f"selected {len(stems)} model(s); this ensemble takes "
            f"{min_models}-{max_models} ({', '.join(stems) or 'none'})."
        )
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        raise SystemExit(f"duplicate model(s) in selection: {dupes}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def apply_transform(scores, transform):
    """Per-column transform. Row-independent by construction — see module docstring."""
    if transform == "raw":
        return scores
    if transform == "logit":
        return pd.DataFrame(
            _logit(scores.to_numpy(dtype=float)), index=scores.index, columns=scores.columns
        )
    raise SystemExit(
        f"unknown transform '{transform}' (choose from {', '.join(TRANSFORMS)}). "
        "ensemble.py's 'rank' is intentionally unavailable: it leaks across the "
        "train/val split and cannot be computed for a single image."
    )


def _score_matrix(dataset_name, stems):
    """One dataset's aligned scores. Rows = images scored by every selected model."""
    frames = []
    resize_tags = {}
    for stem in stems:
        path = csv_path(dataset_name, stem)
        if not os.path.isfile(path):
            raise SystemExit(
                f"no CSV for model '{stem}' on dataset '{dataset_name}': {path}"
            )
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        missing = {"filepath", "score"} - set(df.columns)
        if missing:
            raise SystemExit(f"{path} is missing column(s) {sorted(missing)}")
        if "resize" in df.columns and len(df):
            resize_tags[stem] = str(df["resize"].iloc[0])
        frames.append(df.set_index("filepath")["score"].rename(stem))

    distinct = set(resize_tags.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{s}={resize_tags[s]}" for s in stems if s in resize_tags)
        raise SystemExit(
            f"refusing to combine across mixed resize regimes on '{dataset_name}' "
            f"({detail}). Re-run the odd models at a matching --resize, or select "
            "only same-regime models."
        )

    scores = pd.concat(frames, axis=1, join="inner").sort_index()
    if scores.empty:
        raise SystemExit(
            f"no images scored by all of {stems} on '{dataset_name}' (inner join is empty)."
        )
    return scores[list(stems)], (distinct.pop() if distinct else "unknown")


@dataclass
class Pooled:
    """A fittable (X, y) plus everything needed to interpret or reproduce it."""

    X: pd.DataFrame          # (dataset, filepath) x stems, transformed
    y: pd.Series             # 0/1 from the dataset name
    groups: pd.Series        # dataset name per row
    stems: list              # column order — part of the fitted model
    transform: str
    resize: str
    datasets: list = field(default_factory=list)

    @property
    def n_features(self):
        return len(self.stems)

    def summary(self):
        rows = []
        for ds in self.datasets:
            mask = (self.groups == ds).to_numpy()
            rows.append({
                "dataset": ds,
                "images": int(mask.sum()),
                "label": int(self.y[mask].iloc[0]) if mask.any() else -1,
            })
        return pd.DataFrame(rows)


def build_pooled(datasets, stems, transform="raw"):
    """Stack every dataset's aligned scores into one labeled (X, y)."""
    validate_selection(stems)

    frames = []
    resize_tags = {}
    for ds in datasets:
        scores, tag = _score_matrix(ds, stems)
        resize_tags[ds] = tag
        block = scores.copy()
        block.index.name = "filepath"
        block = block.reset_index()
        block.insert(0, "dataset", ds)
        block["label"] = label_for_dataset(ds)
        frames.append(block)

    distinct = set(resize_tags.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{d}={resize_tags[d]}" for d in datasets)
        raise SystemExit(
            f"refusing to pool across mixed resize regimes ({detail}). The booster "
            "would learn the preprocessing difference as if it were signal — here "
            "the regime is perfectly confounded with the label. Re-run the odd "
            "datasets at a matching --resize."
        )
    resize = distinct.pop() if distinct else "unknown"

    pooled = pd.concat(frames, ignore_index=True).set_index(["dataset", "filepath"]).sort_index()

    y = pooled["label"].astype(int)
    if y.nunique() < 2:
        only = "real (0)" if int(y.iloc[0]) == 0 else "fake (1)"
        raise SystemExit(
            f"pooled set is single-class — every selected dataset is {only}. "
            "Gradient boosting is supervised: include at least one ffhq* dataset "
            "(label 0) and one generator dataset (label 1). For a label-free "
            "combine use ensemble.py instead."
        )

    X = apply_transform(pooled[list(stems)], transform)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise SystemExit(
            f"non-finite values in the design matrix after transform (clamp eps={_EPS}); "
            "check the source CSVs for NaN scores."
        )

    return Pooled(
        X=X,
        y=y,
        groups=pooled.index.get_level_values("dataset").to_series(index=pooled.index),
        stems=list(stems),
        transform=transform,
        resize=resize,
        datasets=list(datasets),
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def shard_groups(index):
    """'<dataset>/<shard>' per row, from the (dataset, filepath) MultiIndex.

    run.py writes 'shard_01/00042.png', so the leading component is the shard. A
    path with no directory component falls back to the dataset as its own group,
    degrading by_shard to by_dataset for those rows rather than inventing groups.
    """
    out = []
    for ds, path in zip(index.get_level_values("dataset"), index.get_level_values("filepath")):
        head = str(path).replace("\\", "/").split("/")[0]
        out.append(f"{ds}/{head}" if head and head != str(path) else str(ds))
    return pd.Series(out, index=index)


def make_split(pooled, mode="random", val_frac=0.2, seed=0, holdout_datasets=None):
    """Boolean train/val masks aligned to pooled.X's index.

    The mode decides which question the validation number answers:

      random      stratified rows. Does a booster help at all? Optimistic —
                  images from one shard are near-siblings and land on both sides.
      by_shard    whole shard_XX dirs, stratified per dataset. Does it generalize
                  past the exact images it saw?
      by_dataset  whole datasets. Does it transfer to a generator never seen?
    """
    n = len(pooled.X)
    index = pooled.X.index
    y = pooled.y.to_numpy()

    if mode == "by_dataset":
        holdout = list(holdout_datasets or [])
        if not holdout:
            raise SystemExit(
                "by_dataset needs at least one --holdout-dataset "
                f"(available: {', '.join(pooled.datasets)})."
            )
        unknown = [d for d in holdout if d not in pooled.datasets]
        if unknown:
            raise SystemExit(f"--holdout-dataset {unknown} not in the pooled set.")
        if set(holdout) == set(pooled.datasets):
            raise SystemExit("cannot hold out every dataset — nothing left to train on.")
        val_mask = pooled.groups.isin(holdout).to_numpy()

    elif mode == "by_shard":
        rng = np.random.default_rng(seed)
        shard_arr = shard_groups(index).to_numpy()
        dataset_arr = pooled.groups.to_numpy()
        val_mask = np.zeros(n, dtype=bool)
        # Stratified per DATASET, not per class. Every dataset is single-class, so
        # this stratifies by class for free, and it additionally guarantees each
        # dataset appears on both sides — otherwise a class's held-out shards can
        # all come from one generator, leaving another absent from validation and
        # its row missing from the per-dataset breakdown.
        for ds in pooled.datasets:
            ds_shards = pd.unique(shard_arr[dataset_arr == ds])
            if len(ds_shards) < 2:
                raise SystemExit(
                    f"dataset '{ds}' has only {len(ds_shards)} shard(s) "
                    f"({', '.join(map(str, ds_shards))}), so a shard-grouped split cannot "
                    "hold any out while leaving something to train on. Upload more shards "
                    "or use --split random."
                )
            order = rng.permutation(len(ds_shards))
            k = max(1, min(len(ds_shards) - 1, int(round(val_frac * len(ds_shards)))))
            val_mask |= np.isin(shard_arr, list(ds_shards[order[:k]]))

    elif mode == "random":
        if not 0.0 < val_frac < 1.0:
            raise SystemExit(f"--val-frac must be in (0, 1), got {val_frac}")
        rng = np.random.default_rng(seed)
        val_mask = np.zeros(n, dtype=bool)
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            val_mask[idx[:max(1, int(round(val_frac * len(idx))))]] = True

    else:
        raise SystemExit(f"unknown split '{mode}' (choose from {', '.join(SPLITS)})")

    train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise SystemExit("split produced an empty side; adjust --val-frac / --holdout-dataset.")
    if len(np.unique(y[train_mask])) < 2:
        raise SystemExit(
            "training side is single-class after the split. With by_dataset, keep at "
            "least one ffhq* and one generator dataset out of the holdout — and note "
            "that with only one real source, by_shard is the split that can hold out "
            "real images without starving training of them."
        )

    return pd.Series(train_mask, index=index), pd.Series(val_mask, index=index)


def split_warnings(pooled, val_mask):
    """Non-fatal things worth printing about a split."""
    warns = []
    val_y = pooled.y[val_mask.to_numpy()]
    if val_y.nunique() < 2:
        only = "real (0)" if int(val_y.iloc[0]) == 0 else "fake (1)"
        warns.append(
            f"validation side is single-class ({only}): accuracy there is just that "
            "class's recall and AUROC is undefined. Hold out one ffhq* AND one "
            "generator dataset, or use --split by_shard."
        )
    return warns


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def sample_weights(y, mode="balanced"):
    """Per-row weights giving each class equal total mass (or all ones).

    Real images come from ffhq alone while fakes come from every generator, so
    the pool's balance tracks how many generator shards happen to be uploaded —
    collection order, not a real prior. Passed to the booster's fit() as
    sample_weight so it cannot score well by answering 'fake' for everything.
    """
    y = np.asarray(y)
    if mode == "none":
        return np.ones(len(y), dtype=float)
    if mode != "balanced":
        raise SystemExit(
            f"unknown class weight '{mode}' (choose from {', '.join(CLASS_WEIGHTS)})"
        )
    weights = np.ones(len(y), dtype=float)
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        weights[y == cls] = len(y) / (len(classes) * count)
    return weights
