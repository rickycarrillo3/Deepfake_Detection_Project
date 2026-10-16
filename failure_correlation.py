#!/usr/bin/env python3
"""
failure_correlation.py

Reads every prediction CSV directly in results/ (flat — no per-model
subfolders), named:

    <dataset>_<stem>_predictions.csv

where <stem> is whatever a model's result_stem() returned (e.g. "aide_DM",
"co-spy_GAN") or, for a model that doesn't define result_stem(), just its
plain model name (e.g. "cnndetection", "DMimageDetection_DM"). <stem> is used
directly as the column/model identifier below — it's already authoritative
(it's literally what model.py named the file), so there's no separate
lookup table to keep in sync or accidentally let go stale.

Groups CSVs by dataset, computes model failure correlations, and writes
plots + CSVs into

results/plots/<dataset>/

Datasets here are a MIX of real and fake DATASETS (not a mix within a single
CSV) — ffhq is entirely real, every other dataset is entirely fake, per
loaders.label_for_dataset. The CSVs themselves have no label column at all
(run.py deliberately doesn't write one), so every metric below derives label
from which dataset a file came from, not from a per-row value:

  delta_vector_<model> = the CSV's own "Predicted label" column, trusted
                          as-is rather than re-thresholding score ourselves
  error_<model>         = abs(label - score)       -- 0 = perfect, 1 = maximally wrong
  correct_<model>       = (delta_vector == label)   -- did this model's decision match the label
  incorrect_<model>     = 1 - correct_<model>       -- used for "failure correlation" (do models
                                                        fail on the same images), NOT raw delta_vector,
                                                        since delta_vector=1 is a success on a fake
                                                        image but a failure on a real one
  oracle accuracy       = fraction of images where AT LEAST ONE model was correct

delta_vector_<model> (raw score >= threshold) is still written to the
combined CSV as a diagnostic, but is no longer what correlations or oracle
accuracy are computed from.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Discovery and ground truth are shared with the other analysis scripts rather
# than re-derived here: ensemble.py anchors the <dataset>_<stem>_predictions.csv
# split on the dataset directory names under data/ (both halves contain
# underscores, so splitting the filename cannot be done reliably), and
# label_for_dataset is the same ffhq-substring rule run.py labelled with.
from ensemble import csv_path, discover_datasets, model_stems_for  # noqa: E402
from loaders.dataset import label_for_dataset  # noqa: E402

###############################################################################
# Configuration
###############################################################################

# Anchored on this file, not the CWD, so the script works from any directory
# (as run.py and the other analysis scripts already do).
RESULTS_DIR = Path(REPO_ROOT) / "results"
OUTPUT_DIR = RESULTS_DIR / "plots"

# dataset_name -> benchmark accuracy to draw as a reference line on that
# dataset's accuracy chart. None / absent -> no line drawn. Keys are the
# dataset directory names under data/, the same names that appear in the CSV
# filenames.
BENCHMARK_LINES = {
    "flux_faces": 0.98,
}


###############################################################################
# Discover CSVs
###############################################################################


def discover_entries():
    """dataset_name -> [(model stem, CSV path)], over the flat results/ dir.

    Both discovery and the dataset->stem filename split come from ensemble.py,
    so all three analysis scripts see exactly the same set of CSVs.
    """
    return {
        dataset: [(stem, Path(csv_path(dataset, stem))) for stem in model_stems_for(dataset)]
        for dataset in discover_datasets()
    }


###############################################################################
# Build combined table
###############################################################################


def build_combined(dataset, entries):

    # Dataset-level ground truth, from the one rule the whole project labels
    # with (ffhq -> real, anything else -> fake). The CSVs carry no label
    # column — run.py deliberately doesn't write one.
    label_value = label_for_dataset(dataset)

    frames = []
    resize_settings = set()

    used = set()

    for stem, csv in sorted(entries):

        if stem in used:
            print(f"Duplicate model '{stem}' (from {csv.name}), skipping.")
            continue

        used.add(stem)

        df = pd.read_csv(csv).set_index("filepath").sort_index()

        missing = {"score", "Predicted label"} - set(df.columns)
        if missing:
            print(f"WARNING: {csv} is missing column(s) {missing} — skipping {stem}.")
            continue

        if "resize" in df.columns:
            resize_settings.update(df["resize"].unique())

        score = df["score"].copy()
        delta = df["Predicted label"].astype(int)   # already thresholded by run.py — trust it, don't re-derive
        label = label_value                          # dataset-level, not per-row: this dataset is all-real or all-fake

        sub = pd.DataFrame(index=df.index)

        sub[f"score_{stem}"] = score
        sub[f"delta_vector_{stem}"] = delta                        # diagnostic only, not used for correlation below
        sub[f"error_{stem}"] = (label - score).abs()                # label-aware: 0=perfect, 1=maximally wrong
        correct = (delta == label).astype(int)
        sub[f"correct_{stem}"] = correct
        sub[f"incorrect_{stem}"] = 1 - correct                      # used for failure correlation

        frames.append(sub)

    if not frames:
        raise ValueError("No usable models for this dataset (see warnings above).")

    combined = pd.concat(frames, axis=1, join="inner").sort_index()
    combined["label"] = label_value

    if len(resize_settings) > 1:
        print(
            f"WARNING: models being combined for {dataset} used different --resize "
            f"settings ({sorted(resize_settings)}) — this is comparing apples to oranges "
            f"unless that's intentional."
        )

    # Remove PatchForensics
    combined = combined.drop(
        columns=[
            c
            for c in combined.columns
            if "patchforensics" in c.lower()
        ],
        errors="ignore",
    )

    return combined


###############################################################################
# Correlations
###############################################################################


def compute_correlations(combined):

    error_cols = [c for c in combined.columns if c.startswith("error_")]

    # Failure correlation uses incorrect_ (label-aware: did this model get
    # this image wrong), NOT delta_vector_ (raw "predicted fake" — which
    # conflates success-on-fake and failure-on-real for a mixed dataset).
    incorrect_cols = [c for c in combined.columns if c.startswith("incorrect_")]

    valid_incorrect = [
        c
        for c in incorrect_cols
        if combined[c].nunique() > 1
    ]

    removed = set(incorrect_cols) - set(valid_incorrect)

    if removed:
        print("\nIgnoring constant incorrect vectors (model always right or always wrong on this dataset):")

        for c in sorted(removed):
            print(" ", c)

    return {
        "error_pearson":
            combined[error_cols].corr(method="pearson"),

        "failure_pearson":
            combined[valid_incorrect].corr(method="pearson"),

        "failure_spearman":
            combined[valid_incorrect].corr(method="spearman"),
    }


###############################################################################
# Plotting
###############################################################################


def plot_heatmap(corr, title, outfile):

    n = len(corr)

    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.1 * n + 2))

    im = ax.imshow(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")

    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)

    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if abs(v) > 0.6 else "black",
            )

    fig.colorbar(im, label="correlation")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_accuracy_rates(combined, title, outfile, benchmark=None):
    """Per-model accuracy (fraction of images where delta_vector matched the
    true label) — replaces the old raw positive-rate chart, which only meant
    "accuracy" for an all-fake dataset."""

    correct_cols = [
        c
        for c in combined.columns
        if c.startswith("correct_")
    ]

    rates = combined[correct_cols].mean().sort_values(ascending=False)

    labels = [
        c.replace("correct_", "")
        for c in rates.index
    ]

    fig, ax = plt.subplots(
        figsize=(0.6 * len(labels) + 3, 5)
    )

    colors = [
        "#c44e52" if r in (0, 1) else "#4c72b0"
        for r in rates
    ]

    bars = ax.bar(labels, rates, color=colors)

    ax.set_ylim(0, 1.05)

    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if benchmark is not None:
        ax.axhline(
            benchmark,
            color="red",
            linestyle=":",
            label=f"Benchmark ({benchmark:.2f})",
        )
        ax.legend()

    for b, r in zip(bars, rates):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r + 0.015,
            f"{r:.2f}",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


###############################################################################
# Oracle accuracy
###############################################################################


def oracle_accuracy(combined):
    """Fraction of images where AT LEAST ONE model was correct (delta_vector
    matched the true label) — the achievable accuracy of an ideal selector
    that always picks a correct model when one exists. No longer excludes
    constant-positive-rate models: that was only a workaround for the old
    "positive == correct" assumption, which correct_<model> no longer makes."""

    correct_cols = [
        c
        for c in combined.columns
        if c.startswith("correct_")
    ]

    if not correct_cols:
        return None

    oracle = combined[correct_cols].any(axis=1)

    return oracle.mean()


###############################################################################
# Main
###############################################################################


def main():

    datasets = discover_entries()
    if not datasets:
        raise SystemExit(
            f"no prediction CSVs found in {RESULTS_DIR}. Run run.py first. (Datasets are "
            f"discovered from the directory names under data/, so that directory must be "
            f"present even when only the finished CSVs are being analysed.)"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Found datasets: {', '.join(sorted(datasets))}\n")

    for dataset, entries in sorted(datasets.items()):

        print("=" * 80)
        print(dataset)
        print("=" * 80)

        # A dataset with nothing to correlate is skipped, not fatal: aeroblade's
        # CSV is always unusable here (its schema carries a raw reconstruction
        # distance and no "Predicted label"), so a results/ dir holding only
        # that for one dataset must not abort the datasets that follow.
        try:
            combined = build_combined(dataset, entries)
        except ValueError as e:
            print(f"[skip] {dataset}: {e}\n")
            continue

        outdir = OUTPUT_DIR / dataset
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"{combined.shape[0]} images")
        print(f"{len(entries)} models")

        combined.to_csv(outdir / "combined_vectors.csv")

        corrs = compute_correlations(combined)

        for name, corr in corrs.items():
            corr.to_csv(outdir / f"{name}.csv")

        plot_heatmap(
            corrs["error_pearson"],
            f"{dataset} — Error Pearson",
            outdir / "error_correlation_pearson.png",
        )

        plot_heatmap(
            corrs["failure_pearson"],
            f"{dataset} — Failure Pearson",
            outdir / "failure_correlation_pearson.png",
        )

        plot_heatmap(
            corrs["failure_spearman"],
            f"{dataset} — Failure Spearman",
            outdir / "failure_correlation_spearman.png",
        )

        plot_accuracy_rates(
            combined,
            f"{dataset} — Accuracy",
            outdir / "accuracy.png",
            benchmark=BENCHMARK_LINES.get(dataset),
        )

        oracle = oracle_accuracy(combined)

        if oracle is not None:
            print(f"Oracle accuracy: {oracle:.2%}")

        print()


if __name__ == "__main__":
    main()