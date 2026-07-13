"""Shared end-of-run console summary.

Lives at the repo root (like run.py) so both runners — the shared one and
aeroblade's bespoke one — report accuracy and inference time identically. If
each computed its own, the threshold or the rounding would drift apart and the
numbers would stop being comparable across models.

Console only: nothing here is written to a CSV. See CLAUDE.md — the models load
and batch differently, so timings are not comparable enough to record as data.
"""

# A score is a predicted fake-probability in [0, 1]; >= this counts as fake (1).
THRESHOLD = 0.5


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
