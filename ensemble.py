#!/usr/bin/env python3
"""Ensemble experiment harness over the flat results/ CSVs.

Combines the per-model fake-probability scores written by run.py
(results/<dataset>_<model>_<family>_predictions.csv) into an ensemble and
reports, per dataset, whether the ensemble beats the individual detectors.

Three combination rules to start (all in metrics.THRESHOLD terms so the cutoff
can't drift from the rest of the project):

  mean     row-wise mean of the (normalized) scores, thresholded. On --norm raw
           this is plain score averaging.
  logodds  average of the per-model logits (log-odds), squashed back through a
           sigmoid, thresholded at 0.5. Scores are clamped off {0, 1} first so a
           model that emits an exact 0.0 (freqnet does) doesn't send a logit to
           -inf. This is the geometric-mean-of-odds ensemble.
  hard_or  OR of each model's own binary call (score >= metrics.THRESHOLD): the
           ensemble says fake if ANY member does. Mirrors the any(axis=1) oracle
           in failure_correlation.py.

Score comparability. Every model already emits a fake-probability in [0, 1], but
the underlying logits sit on different scales and saturate differently, so a
naive combine is dominated by the most over-confident model. --norm applies a
per-model column transform before combining:

  raw    (default) identity — values stay probabilities, so the 0.5 threshold
         keeps its meaning and per-dataset accuracy is directly interpretable.
  rank   per-model rank mapped to [0, 1].
  zlogit per-model standardized logit (logit(clip(p)) then z-score).

rank/zlogit change what the 0.5 cut means, so they are exploratory for now (to
measure whether calibration matters / to feed the correlation work); their
accuracy is reported against --threshold and flagged as regime-dependent. raw is
the evaluatable default.

Evaluation is per dataset. Each dataset present is single-class — ground truth
comes from the dataset name via loaders.label_for_dataset (ffhq -> 0/real, else
-> 1/fake), the same rule run.py uses — so "accuracy" here is agreement with that
dataset's constant label, and the positive rate is the fraction the model/ensemble
called fake. No cross-dataset pooling and no AUROC (needs both classes) in this
pass.

Aeroblade is out of scope: it has a bespoke runner, extra per-autoencoder
columns, and its own results subdir, so it is not one of these flat CSVs.

Depends only on pandas + numpy (both present wherever pandas is), so it runs
under any model venv or a plain interpreter, like the other analysis scripts.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loaders.dataset import label_for_dataset  # noqa: E402
from metrics import THRESHOLD  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")

_SUFFIX = "_predictions.csv"
# Clamp probabilities off the open interval before logit(): a model that emits an
# exact 0.0 or 1.0 (freqnet does) would otherwise map to +/-inf and blow up both
# the log-odds combine and the zlogit normalization.
_EPS = 1e-6

METHODS = ("mean", "logodds", "hard_or")
NORMS = ("raw", "rank", "zlogit")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_datasets():
    """Dataset names that have at least one flat prediction CSV in results/."""
    datasets = set()
    if not os.path.isdir(DATA_DIR):
        return []
    for name in sorted(os.listdir(DATA_DIR)):
        if os.path.isdir(os.path.join(DATA_DIR, name)) and model_stems_for(name):
            datasets.add(name)
    return sorted(datasets)


def model_stems_for(dataset_name):
    """Model stems ('<model>_<family>') that have a CSV for this dataset.

    The prefix is anchored on the known dataset name rather than split on '_',
    since both dataset and model names contain underscores (ffhq_test,
    grag2021_latent_DM, co-spy_sd-v1_4_DM, ...). A file is
    results/<dataset>_<stem>_predictions.csv; strip the two ends.
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


# ---------------------------------------------------------------------------
# Aligned score matrix
# ---------------------------------------------------------------------------


def build_score_matrix(dataset_name, stems):
    """Read each model's CSV, align on filepath, return (scores_df, resize_tag).

    scores_df: rows = images scored by *every* selected model (inner join, as in
    failure_correlation.build_combined), columns = model stems, values = score.
    Also verifies the models were run under one resize regime — mixing regimes is
    exactly what run.py's `resize` column exists to prevent.
    """
    frames = []
    resize_tags = {}
    for stem in stems:
        path = csv_path(dataset_name, stem)
        if not os.path.isfile(path):
            raise SystemExit(f"no CSV for model '{stem}' on dataset '{dataset_name}': {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "resize" in df.columns and len(df):
            resize_tags[stem] = str(df["resize"].iloc[0])
        col = df.set_index("filepath")["score"].rename(stem)
        frames.append(col)

    distinct = set(resize_tags.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{s}={resize_tags[s]}" for s in stems if s in resize_tags)
        raise SystemExit(
            "refusing to ensemble across mixed resize regimes "
            f"({detail}). Re-run the odd models at a matching --resize, or pass "
            "only same-regime models via --models."
        )
    resize_tag = distinct.pop() if distinct else "unknown"

    scores = pd.concat(frames, axis=1, join="inner").sort_index()
    if scores.empty:
        raise SystemExit(
            f"no images scored by all of {stems} on '{dataset_name}' "
            "(inner join is empty)."
        )
    return scores, resize_tag


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def normalize(scores, norm):
    """Apply a per-model (per-column) transform before combining."""
    if norm == "raw":
        return scores
    if norm == "rank":
        # Average-rank -> [0, 1]. A single-row column maps to 0.5.
        return scores.rank(method="average").sub(1).div(max(len(scores) - 1, 1))
    if norm == "zlogit":
        z = scores.apply(lambda c: pd.Series(_logit(c.to_numpy()), index=c.index))
        std = z.std(ddof=0).replace(0, 1.0)
        return (z - z.mean()) / std
    raise SystemExit(f"unknown --norm '{norm}' (choose from {', '.join(NORMS)})")


# ---------------------------------------------------------------------------
# Ensemble methods -> a combined per-image score column
# ---------------------------------------------------------------------------


def ensemble_scores(normed, raw_scores, method):
    """Combined per-image score for one method.

    `normed` is post-normalization (used by mean); `raw_scores` are the original
    probabilities (used by logodds' clamp and hard_or's own-threshold vote), so
    those two keep their probability semantics regardless of --norm.
    """
    if method == "mean":
        return normed.mean(axis=1)
    if method == "logodds":
        mean_logit = _logit(raw_scores.to_numpy()).mean(axis=1)
        return pd.Series(1.0 / (1.0 + np.exp(-mean_logit)), index=raw_scores.index)
    if method == "hard_or":
        # Union of each model's own binary call; the resulting 0/1 series is
        # itself thresholded at 0.5 downstream, so a 1 stays fake.
        return (raw_scores >= THRESHOLD).any(axis=1).astype(float)
    raise SystemExit(f"unknown --method '{method}' (choose from {', '.join(METHODS)})")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(pred_scores, truth, threshold):
    """(accuracy vs constant ground-truth, positive/fake-predicted rate)."""
    pred = (pred_scores >= threshold).astype(int)
    accuracy = float((pred == truth).mean())
    positive_rate = float(pred.mean())
    return accuracy, positive_rate


def run_dataset(dataset_name, stems, methods, norm, threshold, out_dir):
    scores, resize_tag = build_score_matrix(dataset_name, stems)
    truth = label_for_dataset(dataset_name)
    normed = normalize(scores, norm)

    print("=" * 72)
    print(f"{dataset_name}  ({len(scores)} images, {len(stems)} models, "
          f"resize={resize_tag}, ground-truth label={truth})")
    print(f"norm={norm}  threshold={threshold}")
    if norm != "raw":
        print("  note: rank/zlogit rescale scores, so the threshold cut and the "
              "accuracy below are regime-dependent, not calibrated probabilities.")
    print("-" * 72)
    print(f"{'model / ensemble':32} {'accuracy':>10} {'fake-rate':>10}")
    print("-" * 72)

    # Individual detectors (baseline). Compared at --threshold on their own raw
    # scores so the ensemble rows below have a like-for-like reference.
    for stem in stems:
        acc, pos = evaluate(scores[stem], truth, threshold)
        print(f"{stem:32} {acc:>10.3f} {pos:>10.3f}")

    print("-" * 72)

    out_cols = {}
    for method in methods:
        combined = ensemble_scores(normed, scores, method)
        acc, pos = evaluate(combined, truth, threshold)
        print(f"{'ensemble:' + method:32} {acc:>10.3f} {pos:>10.3f}")
        out_cols[f"score_{method}"] = combined
        out_cols[f"pred_{method}"] = (combined >= threshold).astype(int)

    print()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out = scores.copy()
        out.columns = [f"score_{c}" for c in out.columns]
        for name, col in out_cols.items():
            out[name] = col
        out_path = os.path.join(out_dir, f"{dataset_name}_ensemble.csv")
        out.to_csv(out_path)
        print(f"Wrote per-image ensemble scores to {out_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", action="append",
        help="dataset name (repeatable). Default: every dataset with result CSVs.",
    )
    parser.add_argument(
        "--models",
        help="comma-separated model stems to ensemble (e.g. freqnet_GAN,rine1_DM). "
        "Default: every model with a CSV for the dataset.",
    )
    parser.add_argument(
        "--method", default=",".join(METHODS),
        help=f"comma-separated ensemble methods. Default: all ({', '.join(METHODS)}).",
    )
    parser.add_argument(
        "--norm", default="raw", choices=NORMS,
        help="per-model score normalization before combining (default: raw).",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"fake/real cutoff on the combined score (default: {THRESHOLD}).",
    )
    parser.add_argument(
        "--out", metavar="DIR",
        help="also write per-image ensemble scores to DIR/<dataset>_ensemble.csv.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the model stems discovered per dataset and exit.",
    )
    args = parser.parse_args()

    datasets = args.dataset or discover_datasets()
    if not datasets:
        raise SystemExit(
            f"no prediction CSVs found in {RESULTS_DIR}. Run run.py first."
        )

    if args.list:
        for dataset_name in datasets:
            stems = model_stems_for(dataset_name)
            print(f"{dataset_name}: {', '.join(stems) if stems else '(none)'}")
        return

    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {', '.join(METHODS)}")

    selected = [m.strip() for m in args.models.split(",")] if args.models else None

    for dataset_name in datasets:
        available = model_stems_for(dataset_name)
        if selected is None:
            stems = available
        else:
            missing = [m for m in selected if m not in available]
            if missing:
                raise SystemExit(
                    f"model(s) {missing} have no CSV for '{dataset_name}'. "
                    f"Available: {', '.join(available) or '(none)'}"
                )
            stems = selected
        if len(stems) < 2:
            print(f"[skip] {dataset_name}: need >=2 models to ensemble, have "
                  f"{len(stems)} ({', '.join(stems) or 'none'}).\n")
            continue
        run_dataset(dataset_name, stems, methods, args.norm, args.threshold, args.out)


if __name__ == "__main__":
    main()
