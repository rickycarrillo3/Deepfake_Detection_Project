#!/usr/bin/env python3
"""End-to-end check of the neural-network ensemble on synthetic scores.

Proves the wiring works — pooling, guard rails, splits, class weighting, fit,
predict, metrics, save/load round-trip — without needing real result CSVs and
without touching the real results/ or data/ directories (everything happens in a
temp dir, with this folder's RESULTS_DIR/DATA_DIR redirected for the duration).

    python3 neural_network/smoke_test.py
    neural_network/venv/bin/python neural_network/smoke_test.py

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
from model import MLPEnsemble, parse_hidden  # noqa: E402

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
    tmp = tempfile.mkdtemp(prefix="nn_smoke_")
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
            pooled = data_mod.build_pooled(datasets, STEMS[:width], transform="logit")
            check(pooled.n_features == width and pooled.X.shape[1] == width,
                  f"{width} models -> {width} features, X={pooled.X.shape}")
            check(pooled.y.nunique() == 2, f"{width}-model pool has both classes")
            check(pooled.X.index.duplicated().sum() == 0,
                  f"{width}-model pool has unique (dataset, filepath) keys")
        logit_pooled = data_mod.build_pooled(datasets, STEMS[:4], transform="logit")
        check(np.isfinite(logit_pooled.X.to_numpy()).all(),
              "logit features finite despite exact-0.0 scores")

        pooled = data_mod.build_pooled(datasets, STEMS[:4], transform="logit")

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
        y = pooled.y.to_numpy()
        pw = data_mod.pos_weight(y, "balanced")
        expected = float((y == 0).sum()) / float((y == 1).sum())
        check(abs(pw - expected) < 1e-9,
              f"pos_weight = n_real/n_fake = {pw:.4f}")
        check(data_mod.pos_weight(y, "none") == 1.0, "'none' leaves pos_weight at 1.0")

        print("\n[6] architecture and describe() without torch")
        check(parse_hidden("16,8") == (16, 8) and parse_hidden("") == (),
              "hidden spec parses ('16,8' -> (16, 8), '' -> logistic regression)")
        for width in (3, 4, 5):
            net = MLPEnsemble(n_features=width, hidden=(16, 8))
            check(net.layer_sizes() == (width, 16, 8, 1),
                  f"{width} selected models -> {width} input nodes, layers {net.layer_sizes()}")
        net = MLPEnsemble(n_features=4, hidden=(16, 8))
        # 4*16+16 + 16*8+8 + 8*1+1 = 80 + 136 + 9 = 225
        check(net.n_parameters() == 225, f"parameter count {net.n_parameters()} matches by hand")
        text = net.describe()
        check("sigmoid" in text and "BCEWithLogitsLoss" in text,
              "describe() works unfitted (this is what --dry-run prints)")

        print("\n[7] AUROC without scikit-learn")
        rng = np.random.default_rng(7)
        y_demo = rng.integers(0, 2, size=500)
        s_demo = np.clip(0.5 + 0.25 * (y_demo * 2 - 1) + rng.normal(0, 0.3, 500), 0, 1)
        s_demo[:40] = 0.5                      # ties, to exercise average ranks
        ours = evaluation.auroc(y_demo, s_demo)
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            check(0.0 <= ours <= 1.0, f"rank-based AUROC {ours:.4f} in range (sklearn absent)")
        else:
            check(abs(ours - roc_auc_score(y_demo, s_demo)) < 1e-9,
                  f"rank-based AUROC {ours:.4f} matches scikit-learn exactly")
        check(evaluation.auroc(np.ones(10, dtype=int), np.linspace(0, 1, 10)) is None,
              "AUROC returns None on a single-class set instead of raising")

        print("\n[8] train / predict / persist")
        try:
            import torch  # noqa: F401
        except ImportError:
            print("  SKIP torch not installed — the training loop was NOT exercised.")
            print("       bash neural_network/setup.sh, or run this under a model venv:")
            print("       models/freqnet/venv/bin/python neural_network/smoke_test.py")
        else:
            X_tr, y_tr = pooled.X[tr.to_numpy()], pooled.y[tr.to_numpy()]
            X_va, y_va = pooled.X[va.to_numpy()], pooled.y[va.to_numpy()]
            net = MLPEnsemble(n_features=pooled.n_features, hidden=(16, 8), epochs=80,
                              patience=15, pos_weight=data_mod.pos_weight(y_tr.to_numpy()))
            net.fit(X_tr, y_tr, eval_set=(X_va.to_numpy(), y_va.to_numpy()))
            prob = net.predict_proba(X_va)
            check(prob.shape == (len(X_va),) and 0 <= prob.min() and prob.max() <= 1,
                  f"{len(prob)} sigmoid outputs in [0, 1]")
            check(len(net.history) > 0 and net.best_epoch is not None,
                  f"trained {len(net.history)} epoch(s), best epoch {net.best_epoch}")
            check(net.history[-1]["train_loss"] < net.history[0]["train_loss"],
                  "training loss decreased from the first epoch to the last")
            check(net.mean_ is not None and len(net.mean_) == pooled.n_features,
                  "input scaler fitted on the training rows only, one stat per input node")
            m = evaluation.evaluate(y_va, prob, pooled.groups[va.to_numpy()])
            check(m["accuracy"] > 0.5, f"val accuracy {m['accuracy']:.3f} beats chance")
            check(m["auroc"] is not None, f"AUROC {m['auroc']:.3f} computable")
            check(m["per_dataset"] and len(m["per_dataset"]) == len(datasets),
                  "per-dataset breakdown present")

            out = os.path.join(tmp, "artifacts")
            net.save(out)
            net.write_history(out)
            check(os.path.isfile(os.path.join(out, "training_history.json")),
                  "training history written next to the checkpoint")
            reloaded = MLPEnsemble.load(out)
            check(np.allclose(reloaded.predict_proba(X_va), prob, atol=1e-5),
                  "save/load round-trip reproduces predictions (weights + frozen scaler)")

            for width in (3, 5):
                small = data_mod.build_pooled(datasets, STEMS[:width], transform="logit")
                tiny = MLPEnsemble(n_features=width, hidden=(16, 8), epochs=5)
                tiny.fit(small.X, small.y)
                check(tiny.predict_proba(small.X).shape == (len(small.X),),
                      f"{width}-input network trains and predicts end to end")
            plain = MLPEnsemble(n_features=pooled.n_features, hidden=(), epochs=20)
            plain.fit(X_tr, y_tr)
            check(plain.n_parameters() == pooled.n_features + 1,
                  "empty --hidden gives logistic regression on the scores")

        print("\n[9] artifact contract")
        cfg = {"stems": STEMS[:4], "transform": "logit"}
        evaluation.check_feature_contract(cfg, STEMS[:4], "logit")
        print("  ok   matching stems + transform accepted")
        expect_exit(lambda: evaluation.check_feature_contract(cfg, STEMS[:4][::-1], "logit"),
                    "reordered stems rejected (column order is part of the model)")
        expect_exit(lambda: evaluation.check_feature_contract(cfg, STEMS[:4], "raw"),
                    "mismatched transform rejected")

        print("\nall checks passed\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
