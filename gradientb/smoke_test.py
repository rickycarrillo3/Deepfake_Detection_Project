#!/usr/bin/env python3
"""End-to-end check of the gradient-boosting ensemble on synthetic scores.

Proves the wiring works — pooling, guard rails, splits, class weighting, fit,
predict, metrics, save/load round-trip — without needing real result CSVs and
without touching the real results/ or data/ directories (everything happens in a
temp dir, with this folder's RESULTS_DIR/DATA_DIR redirected for the duration).

    python3 gradient_boosting/smoke_test.py
    gradient_boosting/venv/bin/python gradient_boosting/smoke_test.py

This is a wiring test, not a benchmark. The synthetic detectors are noisy by
construction, so the accuracies mean nothing beyond "finite and better than
chance".
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FOLDER)
for path in (FOLDER, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import data as data_mod  # noqa: E402
import evaluation  # noqa: E402
from model import GradientBoostingEnsemble  # noqa: E402

STEMS = ["fakenet_GAN", "fakenet2_DM", "fakenet3_DM", "fakenet4_GAN", "fakenet5_DM"]
DATASETS = [("ffhq_smoke", 0, 4), ("gen_a_smoke", 1, 3), ("gen_b_smoke", 1, 3)]


def write_synthetic(results_dir, data_dir, seed=0):
    """CSVs matching run.py's exact schema: filepath, score, Predicted label, resize."""
    rng = np.random.default_rng(seed)
    os.makedirs(results_dir, exist_ok=True)
    for name, truth, n_shards in DATASETS:
        os.makedirs(os.path.join(data_dir, name), exist_ok=True)
        paths = [f"shard_{s:02d}/{i:05d}.png" for s in range(1, n_shards + 1) for i in range(120)]
        n = len(paths)
        for k, stem in enumerate(STEMS):
            skill = 0.5 + 0.15 * k
            score = (
                rng.beta(2 + 4 * skill, 6, size=n) if truth == 0
                else rng.beta(6, 2 + 4 * skill, size=n)
            )
            score = np.clip(score, 0.0, 1.0)
            if k == 0:
                score[rng.random(n) < 0.1] = 0.0   # freqnet emits exact 0.0
            pd.DataFrame({
                "filepath": paths,
                "score": score,
                "Predicted label": (score >= 0.5).astype(int),
                "resize": "none",
            }).to_csv(os.path.join(results_dir, f"{name}_{stem}_predictions.csv"), index=False)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok   {message}")


def expect_exit(fn, message):
    try:
        fn()
    except SystemExit:
        print(f"  ok   {message}")
    else:
        raise AssertionError(f"expected SystemExit: {message}")


def main():
    tmp = tempfile.mkdtemp(prefix="gb_smoke_")
    try:
        results_dir = os.path.join(tmp, "results")
        data_mod.RESULTS_DIR = results_dir
        data_mod.DATA_DIR = os.path.join(tmp, "data")
        write_synthetic(results_dir, data_mod.DATA_DIR)

        datasets = [name for name, _, _ in DATASETS]

        print("\n[1] discovery")
        check(set(data_mod.common_stems(datasets)) == set(STEMS),
              f"found all {len(STEMS)} stems across {len(datasets)} datasets")
        check(data_mod.discover_datasets() == sorted(datasets),
              "discover_datasets() finds every dataset with CSVs")

        print("\n[2] pooling at 3, 4 and 5 models")
        for width in (3, 4, 5):
            pooled = data_mod.build_pooled(datasets, STEMS[:width], transform="raw")
            check(pooled.n_features == width and pooled.X.shape[1] == width,
                  f"{width} models -> {width} features, X={pooled.X.shape}")
            check(pooled.y.nunique() == 2, f"{width}-model pool has both classes")
            check(pooled.X.index.duplicated().sum() == 0,
                  f"{width}-model pool has unique (dataset, filepath) keys")
        logit_pooled = data_mod.build_pooled(datasets, STEMS[:4], transform="logit")
        check(np.isfinite(logit_pooled.X.to_numpy()).all(),
              "logit features finite despite exact-0.0 scores")

        pooled = data_mod.build_pooled(datasets, STEMS[:4], transform="raw")

        print("\n[3] splits")
        for mode, kwargs in [("random", {}), ("by_shard", {}),
                             ("by_dataset", {"holdout_datasets": ["gen_b_smoke"]})]:
            tr, va = data_mod.make_split(pooled, mode=mode, val_frac=0.25, seed=0, **kwargs)
            check(not (tr.to_numpy() & va.to_numpy()).any() and tr.any() and va.any(),
                  f"{mode}: train={int(tr.sum())} val={int(va.sum())}, disjoint and non-empty")
        tr, va = data_mod.make_split(pooled, mode="by_shard", val_frac=0.25, seed=0)
        shards = data_mod.shard_groups(pooled.X.index)
        check(not (set(shards[tr.to_numpy()]) & set(shards[va.to_numpy()])),
              "by_shard: no shard straddles the split")

        print("\n[4] guard rails")
        expect_exit(lambda: data_mod.validate_selection(STEMS[:2]), "2 models rejected (below 3)")
        expect_exit(lambda: data_mod.validate_selection(STEMS + ["extra_DM"]),
                    "6 models rejected (above 5)")
        expect_exit(lambda: data_mod.validate_selection(STEMS[:2] + [STEMS[0]]),
                    "duplicate model in selection rejected")
        expect_exit(lambda: data_mod.build_pooled(["gen_a_smoke", "gen_b_smoke"], STEMS[:3]),
                    "single-class pool rejected")
        expect_exit(lambda: data_mod.apply_transform(pooled.X, "rank"), "rank transform rejected")
        expect_exit(lambda: data_mod.make_split(pooled, mode="by_dataset"),
                    "by_dataset without a holdout rejected")
        expect_exit(
            lambda: data_mod.make_split(pooled, mode="by_dataset",
                                        holdout_datasets=["gen_a_smoke", "gen_b_smoke"]),
            "holdout leaving training single-class rejected",
        )
        # Mixed resize regimes across datasets.
        bad = os.path.join(results_dir, f"ffhq_smoke_{STEMS[0]}_predictions.csv")
        original = pd.read_csv(bad)
        for stem in STEMS[:4]:
            path = os.path.join(results_dir, f"ffhq_smoke_{stem}_predictions.csv")
            df = pd.read_csv(path)
            df["resize"] = "256x256"
            df.to_csv(path, index=False)
        expect_exit(lambda: data_mod.build_pooled(datasets, STEMS[:4]),
                    "mixed resize regimes across datasets rejected")
        for stem in STEMS[:4]:
            path = os.path.join(results_dir, f"ffhq_smoke_{stem}_predictions.csv")
            df = pd.read_csv(path)
            df["resize"] = "none"
            df.to_csv(path, index=False)
        original.to_csv(bad, index=False)

        print("\n[5] class weighting")
        weights = data_mod.sample_weights(pooled.y.to_numpy(), "balanced")
        y = pooled.y.to_numpy()
        check(abs(weights[y == 1].sum() - weights[y == 0].sum()) < 1e-6,
              "balanced weights give each class equal total mass")
        check((data_mod.sample_weights(y, "none") == 1).all(), "'none' leaves weights at 1")

        print("\n[6] describe() without scikit-learn")
        est = GradientBoostingEnsemble(n_features=pooled.n_features)
        text = est.describe()
        check("gradient boosting" in text and "P(fake)" in text,
              "describe() works unfitted (this is what --dry-run prints)")

        print("\n[7] fit / predict / persist")
        try:
            import sklearn  # noqa: F401
        except ImportError:
            print("  SKIP scikit-learn not installed — the fit path was NOT exercised.")
            print("       bash gradient_boosting/setup.sh, then re-run.")
        else:
            X_tr, y_tr = pooled.X[tr.to_numpy()], pooled.y[tr.to_numpy()]
            X_va, y_va = pooled.X[va.to_numpy()], pooled.y[va.to_numpy()]
            w = data_mod.sample_weights(y_tr.to_numpy(), "balanced")
            for backend in ("sklearn_gb", "sklearn_hist"):
                est = GradientBoostingEnsemble(
                    n_features=pooled.n_features, backend=backend, n_estimators=40
                )
                est.fit(X_tr, y_tr, sample_weight=w)
                prob = est.predict_proba(X_va)
                check(prob.shape == (len(X_va),) and 0 <= prob.min() and prob.max() <= 1,
                      f"{backend}: {len(prob)} probabilities in [0, 1]")
                m = evaluation.evaluate(y_va, prob, pooled.groups[va.to_numpy()])
                check(m["accuracy"] > 0.5, f"{backend}: val accuracy {m['accuracy']:.3f} beats chance")
                check(m["auroc"] is not None,
                      f"{backend}: AUROC {m['auroc']:.3f} computable (pooled = both classes)")
                check(m["per_dataset"] and len(m["per_dataset"]) == len(datasets),
                      f"{backend}: per-dataset breakdown present")
                out = os.path.join(tmp, f"artifacts_{backend}")
                est.save(out)
                check(np.allclose(GradientBoostingEnsemble.load(out).predict_proba(X_va), prob),
                      f"{backend}: save/load round-trip reproduces predictions")
            imp = est.feature_importance()
            check(imp is None or set(imp) == set(pooled.stems),
                  "feature importance is keyed by detector name (or absent for hist)")

        print("\n[8] artifact contract")
        cfg = {"stems": STEMS[:4], "transform": "raw"}
        evaluation.check_feature_contract(cfg, STEMS[:4], "raw")
        print("  ok   matching stems + transform accepted")
        expect_exit(lambda: evaluation.check_feature_contract(cfg, STEMS[:4][::-1], "raw"),
                    "reordered stems rejected (column order is part of the model)")
        expect_exit(lambda: evaluation.check_feature_contract(cfg, STEMS[:4], "logit"),
                    "mismatched transform rejected")

        print("\nall checks passed\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
