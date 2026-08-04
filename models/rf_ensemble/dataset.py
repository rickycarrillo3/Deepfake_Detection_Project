"""
Builds training examples for the random-forest ensemble purely from
already-computed results, WITHOUT running any underlying model's own code --
same approach as models/gated_ensemble/dataset.py.

Each example is (expert_score_vector, label) for one image:
  - expert_score_vector: one score per chosen model (the "experts" you pick),
    read from results/<dataset>_<stem>_predictions.csv
  - label: derived from the DATASET name, not a per-row column (ffhq_faces is
    all-real, every other dataset is all-fake -- matches failure_correlation.py
    and models/gated_ensemble's convention)

If a dataset hasn't been scored yet by one of the experts you pick, its
images simply aren't usable -- run.py for that model/dataset combination
first.
"""
import os

import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(MODEL_DIR))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Exact dataset directory names (under data/) that are entirely real images.
# Everything else is treated as entirely fake.
REAL_DATASETS = {"ffhq_faces"}


def label_for_dataset(dataset_name):
    return 0 if dataset_name in REAL_DATASETS else 1


def load_score_tables(dataset_name, stems):
    """dataset_name -> {stem: DataFrame indexed by filepath}"""
    tables = {}
    for stem in stems:
        path = os.path.join(RESULTS_DIR, f"{dataset_name}_{stem}_predictions.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path} -- run the model that produces stem '{stem}' "
                f"on dataset '{dataset_name}' first (via the top-level run.py) "
                f"before training on it."
            )
        tables[stem] = pd.read_csv(path).set_index("filepath")
    return tables


def build_training_table(dataset_names, stems):
    """Returns (X, y, filepaths):
       X: (n_examples, len(stems)) float array of expert scores
       y: (n_examples,) int array of labels
       filepaths: parallel list of (dataset_name, filepath) for traceability

    Inner-joins on filepath across all chosen stems, per dataset -- an image
    missing from even one expert's CSV (e.g. it errored during that model's
    own run) is skipped for that dataset."""
    if not (3 <= len(stems) <= 5):
        raise ValueError(f"pick 3-5 expert models, got {len(stems)}: {stems}")

    rows = []
    labels = []
    keys = []

    for dataset_name in dataset_names:
        tables = load_score_tables(dataset_name, stems)

        common_filepaths = set(tables[stems[0]].index)
        for stem in stems[1:]:
            common_filepaths &= set(tables[stem].index)
        if not common_filepaths:
            print(f"WARNING: no filepaths in common across {stems} for dataset {dataset_name} -- skipping it.")
            continue

        label = label_for_dataset(dataset_name)
        for filepath in sorted(common_filepaths):
            rows.append([float(tables[stem].loc[filepath, "score"]) for stem in stems])
            labels.append(label)
            keys.append((dataset_name, filepath))

    if not rows:
        raise ValueError("No usable training examples found -- check dataset_names/stems.")

    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), keys
