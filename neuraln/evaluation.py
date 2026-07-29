"""Scoring and artifact I/O for the neural-network ensemble.

Named evaluation.py rather than metrics.py on purpose: train.py runs as a
script, so its own folder is first on sys.path and a local metrics.py would
shadow the repo-root metrics.py that supplies THRESHOLD.

Pooling finally makes AUROC meaningful. ensemble.py skips it explicitly ("needs
both classes") because each dataset there is single-class; the pooled matrix has
both, so a threshold-free ranking measure is available — useful precisely
because the 0.5 cutoff is somewhat arbitrary for a combined score, and because a
network trained with pos_weight shifts where its probabilities sit without
necessarily changing how well it ranks.

AUROC is computed here from ranks rather than imported from scikit-learn, so
this folder's only real dependency stays torch. It is a dozen lines and lets
--method-free evaluation run under any model venv.

Per-dataset accuracy is reported alongside the overall figure. With
single-class datasets that breakdown is the informative part: it separates "the
network is good" from "the network answers fake for everything and most of the
pool is fake".
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FOLDER)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metrics import THRESHOLD  # noqa: E402

ARTIFACT_ROOT = os.path.join(REPO_ROOT, "results", "neural_network")
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"
VAL_PRED_NAME = "val_predictions.csv"


def auroc(y_true, score):
    """Area under the ROC curve, via the Mann-Whitney U identity.

    AUROC is the probability that a randomly chosen fake image is ranked above a
    randomly chosen real one, which equals

        (sum of the fakes' ranks - n_fake(n_fake + 1)/2) / (n_fake * n_real)

    Average ranks are used for ties — a network that emits the same probability
    for two images should get half credit for that pair, not full.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(y_true, prob, groups=None, threshold=THRESHOLD):
    """Accuracy / balanced accuracy / precision / recall / F1 / AUROC (+ per dataset)."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    def _safe(num, den):
        return float(num / den) if den else float("nan")

    recall = _safe(tp, tp + fn)
    specificity = _safe(tn, tn + fp)
    precision = _safe(tp, tp + fp)
    f1 = (
        _safe(2 * precision * recall, precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": _safe(tp + tn, len(y_true)),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc(y_true, prob),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

    if groups is not None:
        groups = np.asarray(groups)
        out["per_dataset"] = {
            str(ds): {
                "n": int((groups == ds).sum()),
                "accuracy": _safe(
                    int((pred[groups == ds] == y_true[groups == ds]).sum()),
                    int((groups == ds).sum()),
                ),
                "fake_rate": float(pred[groups == ds].mean()),
            }
            for ds in pd.unique(groups)
        }
    return out


def print_metrics(title, m):
    value = m.get("auroc")
    print(title)
    print(
        f"  n={m['n']}  acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  "
        f"auroc={'n/a' if value is None else format(value, '.4f')}"
    )
    print(
        f"  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
        f"specificity={m['specificity']:.4f}  f1={m['f1']:.4f}"
    )
    per = m.get("per_dataset")
    if per:
        print(f"  {'dataset':28} {'n':>7} {'accuracy':>10} {'fake-rate':>10}")
        for ds, v in per.items():
            print(f"  {ds:28} {v['n']:>7} {v['accuracy']:>10.4f} {v['fake_rate']:>10.4f}")


def run_dir(name, root=ARTIFACT_ROOT):
    return os.path.join(root, name)


def write_artifacts(out_dir, config, metrics=None, val_predictions=None):
    os.makedirs(out_dir, exist_ok=True)
    config = dict(config)
    config.setdefault("written_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with open(os.path.join(out_dir, CONFIG_NAME), "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
    if metrics is not None:
        with open(os.path.join(out_dir, METRICS_NAME), "w") as fh:
            json.dump(metrics, fh, indent=2, sort_keys=True)
    if val_predictions is not None:
        val_predictions.to_csv(os.path.join(out_dir, VAL_PRED_NAME))
    return out_dir


def read_config(out_dir):
    with open(os.path.join(out_dir, CONFIG_NAME)) as fh:
        return json.load(fh)


def check_feature_contract(config, stems, transform):
    """A fitted network is only valid on the exact columns, in the exact order, it saw.

    Feeding the same three detectors in a different order produces confident
    nonsense rather than an error, so the check is explicit. The frozen scaler in
    the checkpoint is per-column too, which is the other half of why order matters.
    """
    saved = list(config.get("stems", []))
    if saved != list(stems):
        raise SystemExit(
            f"feature mismatch: fit on [{', '.join(saved)}] but handed "
            f"[{', '.join(stems)}]. Column order is part of the model."
        )
    if config.get("transform") != transform:
        raise SystemExit(
            f"transform mismatch: fit with '{config.get('transform')}', "
            f"called with '{transform}'."
        )
