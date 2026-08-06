#!/usr/bin/env python3
"""Pooled real-vs-fake AUROC over the flat results/ CSVs.

Every prediction CSV is single-class: ground truth comes from the dataset name
(loaders.label_for_dataset — ffhq -> 0/real, anything else -> 1/fake), so one
CSV alone has no ROC to speak of. This script pairs each fake dataset's CSV with
the ffhq CSV *for the same model*, pools the two score columns, and reports
AUROC alongside the accuracy at metrics.THRESHOLD that the runners already
print.

Pairing is automatic: every dataset with CSVs whose name marks it real (ffhq) is
a real pool, every other dataset is a fake pool, and one row is reported per
(real, fake, model). --real/--fake pin it down when more than one of either
exists.

Class imbalance is expected and fine — 10k fakes against 20k reals is the normal
shape here. AUROC is rank-based (see metrics.auroc), so each fake/real pair
counts equally and the larger pool does not dominate. The accuracy columns are
not imbalance-proof in the same way, which is why the per-side rates are broken
out next to the pooled one:

  auroc        ranking quality over the pooled set; 0.5 is chance, threshold-free
  accuracy     pooled, at --threshold; drifts toward whichever class is larger
  real_acc     fraction of ffhq images called real (true-negative rate)
  fake_acc     fraction of fake images called fake (true-positive rate)

A model is only reported for a pair when it has a CSV on both sides and both
were run under the same --resize regime; mismatches are skipped with a notice
rather than silently pooled, for the reason run.py records the regime at all.

Depends only on pandas (+ ensemble.py's discovery helpers, which pull numpy), so
it runs under any model venv or a plain interpreter, like the other analysis
scripts here.
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ensemble import RESULTS_DIR, csv_path, discover_datasets, model_stems_for  # noqa: E402
from loaders.dataset import label_for_dataset  # noqa: E402
from metrics import THRESHOLD, auroc  # noqa: E402

SUMMARY_CSV = os.path.join(RESULTS_DIR, "auroc_summary.csv")

SUMMARY_COLUMNS = [
    "real_dataset",
    "fake_dataset",
    "model",
    "resize",
    "n_real",
    "n_fake",
    "auroc",
    "accuracy",
    "real_acc",
    "fake_acc",
]


def _load_scores(dataset_name, stem):
    """(scores, resize_tag) for one CSV, or (None, None) if it isn't there."""
    path = csv_path(dataset_name, stem)
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "score" not in df.columns:
        print(f"[skip] {path}: no 'score' column.")
        return None, None
    resize_tag = str(df["resize"].iloc[0]) if "resize" in df.columns and len(df) else "unknown"
    return df["score"].astype(float).tolist(), resize_tag


def evaluate_pair(real_dataset, fake_dataset, stem, threshold):
    """One summary row for a (real dataset, fake dataset, model), or None.

    Returns None when either side is missing a CSV, either is empty, or the two
    were produced under different --resize regimes (pooling those would compare
    scores from two different input pipelines).
    """
    real_scores, real_resize = _load_scores(real_dataset, stem)
    fake_scores, fake_resize = _load_scores(fake_dataset, stem)
    if real_scores is None or fake_scores is None:
        return None
    if not real_scores or not fake_scores:
        print(f"[skip] {stem}: {real_dataset} vs {fake_dataset} — a side has no rows.")
        return None
    if real_resize != fake_resize:
        print(
            f"[skip] {stem}: {real_dataset} (resize={real_resize}) and "
            f"{fake_dataset} (resize={fake_resize}) were run under different "
            f"resize regimes — re-run one to match before pooling them."
        )
        return None

    # label is the dataset's own ground truth, the same rule run.py used; the
    # CSVs deliberately carry no label column of their own.
    rows = [{"score": s, "label": 0} for s in real_scores]
    rows += [{"score": s, "label": 1} for s in fake_scores]

    real_correct = sum(1 for s in real_scores if s < threshold)
    fake_correct = sum(1 for s in fake_scores if s >= threshold)

    return {
        "real_dataset": real_dataset,
        "fake_dataset": fake_dataset,
        "model": stem,
        "resize": real_resize,
        "n_real": len(real_scores),
        "n_fake": len(fake_scores),
        # Both classes are present by construction here, so auroc() cannot
        # return None — that guard is for single-dataset callers.
        "auroc": auroc(rows),
        "accuracy": (real_correct + fake_correct) / len(rows),
        "real_acc": real_correct / len(real_scores),
        "fake_acc": fake_correct / len(fake_scores),
    }


def print_pair(real_dataset, fake_dataset, results):
    print("=" * 78)
    print(f"{fake_dataset} (fake) vs {real_dataset} (real)   threshold={THRESHOLD}")
    print("-" * 78)
    header = f"{'model':30} {'auroc':>8} {'accuracy':>9} {'real_acc':>9} {'fake_acc':>9}"
    print(header)
    print("-" * 78)
    for row in sorted(results, key=lambda r: r["auroc"], reverse=True):
        print(
            f"{row['model']:30} {row['auroc']:>8.3f} {row['accuracy']:>9.3f} "
            f"{row['real_acc']:>9.3f} {row['fake_acc']:>9.3f}"
        )
    n_real = results[0]["n_real"]
    fakes = {r["n_fake"] for r in results}
    print("-" * 78)
    print(
        f"{len(results)} model(s), {n_real} real image(s) vs "
        f"{'/'.join(str(n) for n in sorted(fakes))} fake, resize={results[0]['resize']}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--real", action="append",
        help="real dataset name (repeatable). Default: every ffhq dataset with CSVs.",
    )
    parser.add_argument(
        "--fake", action="append",
        help="fake dataset name (repeatable). Default: every non-ffhq dataset with CSVs.",
    )
    parser.add_argument(
        "--models",
        help="comma-separated model stems to report (e.g. freqnet_GAN,rine1_DM). "
        "Default: every model with a CSV on both sides of a pair.",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"fake/real cutoff for the accuracy columns (default: {THRESHOLD}). "
        "AUROC is threshold-free and ignores this.",
    )
    parser.add_argument(
        "--out", metavar="CSV", default=SUMMARY_CSV,
        help=f"write the summary rows here (default: {SUMMARY_CSV}).",
    )
    parser.add_argument("--no-csv", action="store_true", help="print only, write no summary CSV.")
    args = parser.parse_args()

    datasets = discover_datasets()
    if not datasets:
        raise SystemExit(f"no prediction CSVs found in {RESULTS_DIR}. Run run.py first.")

    reals = args.real or [d for d in datasets if label_for_dataset(d) == 0]
    fakes = args.fake or [d for d in datasets if label_for_dataset(d) == 1]
    if not reals or not fakes:
        raise SystemExit(
            "AUROC needs both classes, but the datasets with CSVs are "
            f"{'all fake' if not reals else 'all real'}: {', '.join(datasets)}. "
            "Score an ffhq dataset and a fake one with the same model first."
        )

    selected = [m.strip() for m in args.models.split(",")] if args.models else None

    summary = []
    for real_dataset in reals:
        for fake_dataset in fakes:
            shared = [s for s in model_stems_for(fake_dataset) if s in model_stems_for(real_dataset)]
            stems = [s for s in shared if s in selected] if selected else shared
            if not stems:
                print(
                    f"[skip] {fake_dataset} vs {real_dataset}: no model has a CSV "
                    f"for both.\n"
                )
                continue
            results = [
                row
                for row in (
                    evaluate_pair(real_dataset, fake_dataset, stem, args.threshold)
                    for stem in stems
                )
                if row is not None
            ]
            if results:
                print_pair(real_dataset, fake_dataset, results)
                summary.extend(results)

    if not summary:
        raise SystemExit("no (real, fake, model) pair could be scored — nothing to report.")

    if not args.no_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(args.out, index=False)
        print(f"Wrote {len(summary)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
