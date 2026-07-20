#!/usr/bin/env python3
"""
failure_correlation.py

Reads every detector CSV in results/, groups them by dataset
(flux, ffhq, stylegan-0.5, ...), computes model failure
correlations, and writes plots + CSVs into

results/plots/<dataset>/
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
    "rine": "DM",
    "rine_1": "GAN1",
    "rine_2": "GAN2",
    "rine_4": "GAN4",
    "rine_ldm": "DM",
}

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

    for csv in RESULTS_DIR.rglob("*.csv"):

        if csv.name.endswith("_summary.csv"):
            continue

        folder = csv.parent.name

        if folder not in TRAINING_TYPE:
            continue

        dataset = csv.stem.replace("_faces", "")

        datasets.setdefault(dataset, []).append((pretty_model_name(folder), folder, csv))

    return datasets


###############################################################################
# Build combined table
###############################################################################


def build_combined(entries):

    frames = []

    used = set()

    for model_name, folder, csv in sorted(entries):

        if model_name in used:
            print(f"Duplicate model '{model_name}', skipping.")
            continue

        used.add(model_name)

        df = pd.read_csv(csv).set_index("filepath").sort_index()

        score = df["score"].copy()

        # Grag outputs are inverted
        if folder == "grag2021":
            score = 1 - score

        sub = pd.DataFrame(index=df.index)

        sub[f"score_{model_name}"] = score
        sub[f"delta_vector_{model_name}"] = (score >= DELTA_THRESHOLD).astype(int)
        sub[f"error_{model_name}"] = 1 - score

        frames.append(sub)

    combined = pd.concat(frames, axis=1, join="inner").sort_index()

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

    delta_cols = [c for c in combined.columns if c.startswith("delta_vector_")]

    valid_delta = [
        c
        for c in delta_cols
        if combined[c].nunique() > 1
    ]

    removed = set(delta_cols) - set(valid_delta)

    if removed:
        print("\nIgnoring constant delta vectors:")

        for c in sorted(removed):
            print(" ", c)

    return {
        "error_pearson":
            combined[error_cols].corr(method="pearson"),

        "delta_pearson":
            combined[valid_delta].corr(method="pearson"),

        "delta_spearman":
            combined[valid_delta].corr(method="spearman"),
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


def plot_delta_rates(combined, title, outfile, benchmark=None):

    delta_cols = [
        c
        for c in combined.columns
        if c.startswith("delta_vector_")
    ]

    rates = combined[delta_cols].mean().sort_values(ascending=False)

    labels = [
        c.replace("delta_vector_", "")
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

    ax.set_ylabel("Positive rate")
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

    delta_cols = [
        c
        for c in combined.columns
        if c.startswith("delta_vector_")
    ]

    delta_cols = [
        c
        for c in delta_cols
        if combined[c].mean() < 1.0
    ]

    if not delta_cols:
        return None

    oracle = combined[delta_cols].any(axis=1)

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
            corrs["delta_pearson"],
            f"{dataset} — Delta Pearson",
            outdir / "delta_correlation_pearson.png",
        )

        plot_heatmap(
            corrs["delta_spearman"],
            f"{dataset} — Delta Spearman",
            outdir / "delta_correlation_spearman.png",
        )

        plot_delta_rates(
            combined,
            f"{dataset} — Positive Rate",
            outdir / "delta_positive_rate.png",
            benchmark=BENCHMARK_LINES.get(dataset),
        )

        oracle = oracle_accuracy(combined)

        if oracle is not None:
            print(f"Oracle accuracy: {oracle:.2%}")

        print()


if __name__ == "__main__":
    main()