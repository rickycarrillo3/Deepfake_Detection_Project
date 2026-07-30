"""Shared end-of-run console summary, plus the append-only run log.

Lives at the repo root (like run.py) so both runners — the shared one and
aeroblade's bespoke one — report accuracy and inference time identically. If
each computed its own, the threshold or the rounding would drift apart and the
numbers would stop being comparable across models.

Two different durations are logged; keep them apart:

  * `predict_seconds` — the model's predict() calls summed on their own, with
    weight loading, dataset walking and CSV writing all excluded. This is the
    model number. Now that one shared runner times every model with the same
    code (see run.py), it is measured uniformly and so is comparable across
    models — but only between rows that ran on the same hardware over the same
    images, since neither is recorded here.

  * `wall_seconds` — the whole invocation end to end, weight loading and CSV
    writing included. Operational: how long the job occupied the machine, for
    planning cluster time. Not a property of the detector.

Both go to run_log.csv, deliberately separate from the predictions CSVs, whose
schema stays purely per-image. print_run_summary() prints the same predict-only
number to the console.
"""

import csv
import os
from datetime import datetime

# A score is a predicted fake-probability in [0, 1]; >= this counts as fake (1).
THRESHOLD = 0.5

# Appended to by every run of every model, from both runners. Under results/
# alongside the predictions it describes, and git-ignored with them — it is a
# record of what this machine has run, not a project artifact to share.
RUN_LOG = "results/run_log.csv"

RUN_LOG_COLUMNS = [
    "timestamp",
    "model",
    "dataset",
    "n_images",
    "predict_seconds",
    "wall_seconds",
]


def accuracy(rows):
    """Fraction of rows whose thresholded score matches the ground-truth label."""
    correct = sum(1 for row in rows if (row["score"] >= THRESHOLD) == bool(row["label"]))
    return correct / len(rows)


def print_run_summary(rows, elapsed_seconds):
    """Print accuracy and inference time.

    `elapsed_seconds` must be the model's prediction calls summed on their own —
    no weight loading, no dataset walking, no CSV writing — so that the number
    reflects the model rather than the runner wrapped around it.
    """
    print(f"Accuracy: {accuracy(rows):.2f}")
    print(f"Time elapsed: {elapsed_seconds:.4f}s")


def append_run_log(repo_root, model_name, dataset_name, n_images, predict_seconds, wall_seconds):
    """Append one row for this run to results/run_log.csv, creating it if needed.

    `model_name` is the MODELS registry name from run.py, so a row joins back to
    the registry (and to the predictions CSV) by name. Called once per
    invocation by both runners — under --all/--baselines/--models each model is
    a separate `run.py --model ...` subprocess, so a sweep naturally appends one
    row per model rather than one for the sweep.

    Opened in append mode and written as a single row so that concurrent runs on
    a cluster interleave rows instead of truncating each other's file. The
    header is written only when this handle finds the file empty, so the first
    of those runs creates it and the rest just append.
    """
    log_path = os.path.join(repo_root, RUN_LOG)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "dataset": dataset_name,
                "n_images": n_images,
                "predict_seconds": round(predict_seconds, 4),
                "wall_seconds": round(wall_seconds, 4),
            }
        )
    print(f"Logged this run to {log_path}")
