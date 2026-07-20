#!/usr/bin/env python3
"""
failure_correlation.py

Reads every detector CSV in results/, groups them by dataset
(flux, ffhq, stylegan-0.5, ...), computes model failure
correlations, and writes plots + CSVs into

results/plots/<dataset>/

Datasets here are a MIX of real (label=0) and fake (label=1) images, so every
metric below is computed relative to the ground-truth label column in each
CSV, not blindly derived from the score alone:

  error_<model>     = abs(label - score)         -- 0 = perfect, 1 = maximally wrong
  correct_<model>    = (delta_vector == label)     -- did this model's decision match the label
  incorrect_<model>  = 1 - correct_<model>         -- used for "failure correlation" (do models
                                                       fail on the same images), NOT raw delta_vector,
                                                       since delta_vector=1 is a success on a fake
                                                       image but a failure on a real one
  oracle accuracy    = fraction of images where AT LEAST ONE model was correct

delta_vector_<model> (raw score >= threshold) is still written to the
combined CSV as a diagnostic, but is no longer what correlations or oracle
accuracy are computed from.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# Configuration
###############################################################################

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "plots"

DELTA_THRESHOLD = 0.5

TRAINING_TYPE = {
    "aide_genimage": "MIX",
    "aide_progan": "GAN",
    "aide_sd14": "DM",
    "cnndetection": "GAN",
    "co-spy_progan": "GAN",
    "co-spy_sd-v1_4": "DM",
    "freqnet": "GAN",
    "grag2021": "GAN",
    "grag2021_latent": "DM",
    "grag2021_progan": "GAN",
    "rine_1": "GAN1",
    "rine_2": "GAN2",
    "rine_4": "GAN4",
    "rine_ldm": "DM",
    # NOTE: bare "rine" deliberately excluded — it collided with rine_ldm's
    # pretty name ("rine (DM)" for both), and being alphabetically first,
    # silently won every "duplicate model" skip in build_combined(), so
    # rine_ldm's actual results were being discarded. If results/rine/ is
    # stale (pre-checkpoint-variant), delete it on disk too.
}

# Benchmark accuracy to compare each dataset's models against (now plotted
# against per-model ACCURACY, not raw positive rate — comparing an accuracy
# figure to positive rate was only valid for an all-fake dataset).
BENCHMARK_LINES = {
    "flux": 0.98,
    "ffhq": None,
    "stylegan-0.5": None,
    "stylegan-0.7": None,
}

###############################################################################
# Model naming
###############################################################################


def pretty_model_name(folder):
    if folder in {"grag2021_progan", "grag2021_latent"}:
        base = "DMimageDetection"

    elif folder in {"co-spy_progan", "co-spy_sd-v1_4"}:
        base = "co-spy"

    elif folder in {"rine_1", "rine_2", "rine_4", "rine_ldm"}:
        base = "rine"

    elif folder in {"aide_genimage", "aide_progan", "aide_sd14"}:
        base = "aide"

    else:
        base = folder

    return f"{base} ({TRAINING_TYPE[folder]})"


###############################################################################
# Discover CSVs
###############################################################################


def discover_datasets():
    datasets = {}
    skipped_folders = set()

    for csv in RESULTS_DIR.rglob("*.csv"):

        if csv.name.endswith("_summary.csv"):
            continue

        folder = csv.parent.name

        if folder not in TRAINING_TYPE:
            skipped_folders.add(folder)
            continue

        dataset = csv.stem.replace("_faces", "")

        datasets.setdefault(dataset, []).append((pretty_model_name(folder), folder, csv))

    if skipped_folders:
        print(
            "Skipping folder(s) not listed in TRAINING_TYPE (add them there if "
            f"they should be included): {', '.join(sorted(skipped_folders))}\n"
        )

    return datasets


###############################################################################
# Build combined table
###############################################################################


def build_combined(entries):

    frames = []
    label_frames = []

    used = set()

    for model_name, folder, csv in sorted(entries):

        if model_name in used:
            print(f"Duplicate model '{model_name}', skipping.")
            continue

        used.add(model_name)

        df = pd.read_csv(csv).set_index("filepath").sort_index()

        if "label" not in df.columns:
            print(f"WARNING: {csv} has no 'label' column — skipping {model_name} (can't score correctness without it).")
            continue

        score = df["score"].copy()
        label = df["label"].astype(int)

        # Grag outputs are inverted
        #        if folder == "grag2021":
        #            score = 1 - score

        sub = pd.DataFrame(index=df.index)

        sub[f"score_{model_name}"] = score
        delta = (score >= DELTA_THRESHOLD).astype(int)
        sub[f"delta_vector_{model_name}"] = delta                       # diagnostic only, not used for correlation below
        sub[f"error_{model_name}"] = (label - score).abs()               # label-aware: 0=perfect, 1=maximally wrong
        correct = (delta == label).astype(int)
        sub[f"correct_{model_name}"] = correct
        sub[f"incorrect_{model_name}"] = 1 - correct                     # used for failure correlation

        frames.append(sub)
        label_frames.append(label.rename(model_name))

    if not frames:
        raise ValueError("No usable models for this dataset (see warnings above).")

    combined = pd.concat(frames, axis=1, join="inner").sort_index()

    # Sanity-check that every model's CSV agrees on the ground-truth label for
    # the same image — they should, since label is a property of the image,
    # not the model. Flag rather than silently trust it if they don't.
    labels_df = pd.concat(label_frames, axis=1, join="inner").sort_index()
    if not labels_df.eq(labels_df.iloc[:, 0], axis=0).all().all():
        mismatched = (~labels_df.eq(labels_df.iloc[:, 0], axis=0).all(axis=1)).sum()
        print(f"WARNING: {mismatched} image(s) have disagreeing labels across model CSVs for this dataset — using the first model's label for those.")
    combined["label"] = labels_df.iloc[:, 0]

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

    datasets = discover_datasets()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Found datasets: {', '.join(sorted(datasets))}\n")

    for dataset, entries in sorted(datasets.items()):

        print("=" * 80)
        print(dataset)
        print("=" * 80)

        outdir = OUTPUT_DIR / dataset
        outdir.mkdir(parents=True, exist_ok=True)

        combined = build_combined(entries)

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