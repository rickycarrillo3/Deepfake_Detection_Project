"""Scoring and artifact I/O for the gradient-boosting ensemble.

Named evaluation.py rather than metrics.py on purpose: train.py runs as a
script, so its own folder is first on sys.path and a local metrics.py would
shadow the repo-root metrics.py that supplies THRESHOLD.

Pooling finally makes AUROC meaningful. ensemble.py skips it explicitly ("needs
both classes") because each dataset there is single-class; the pooled matrix has
both, so a threshold-free ranking measure is available here — useful precisely
because the 0.5 cutoff is somewhat arbitrary for a combined score.

Per-dataset accuracy is reported alongside the overall figure. With
single-class datasets that breakdown is the informative part: it separates "the
booster is good" from "the booster answers fake for everything and most of the
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

ARTIFACT_ROOT = os.path.join(REPO_ROOT, "results", "gradient_boosting")
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"
VAL_PRED_NAME = "val_predictions.csv"


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
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_auc_score  # lazy: --dry-run needs no sklearn

        out["auroc"] = float(roc_auc_score(y_true, prob))
    else:
        out["auroc"] = None

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
    auroc = m.get("auroc")
    print(title)
    print(
        f"  n={m['n']}  acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  "
        f"auroc={'n/a' if auroc is None else format(auroc, '.4f')}"
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
    """A fitted booster is only valid on the exact columns, in the exact order, it saw.

    Feeding the same three detectors in a different order produces confident
    nonsense rather than an error, so the check is explicit.
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
