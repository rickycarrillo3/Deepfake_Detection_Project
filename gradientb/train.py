#!/usr/bin/env python3
"""Train the gradient-boosting ensemble over the per-detector results CSVs.

    # what can I select right now?
    python3 gradient_boosting/train.py --list

    # verify the setup without fitting (needs no scikit-learn)
    python3 gradient_boosting/train.py \
        --models freqnet_GAN,rine1_DM,grag2021_latent_DM \
        --dataset ffhq_test --dataset flux_test --dry-run

    # fit, once you're ready
    python3 gradient_boosting/train.py \
        --models freqnet_GAN,rine1_DM,grag2021_latent_DM,cnndetection \
        --dataset ffhq_test --dataset flux_test --run-name gb_4model

--models takes 3-5 detector stems; they become the booster's features, in the
order given. That order is part of the fitted model and is recorded in the run's
config.json.

TRAINING IS NEVER IMPLICIT. Fitting happens only when --dry-run is absent, and
--dry-run is what to use while wiring this up: it builds the real design matrix,
reports class balance and the split, prints per-detector baselines and the exact
booster configuration, then stops.

Ground truth comes from the dataset name (loaders.label_for_dataset: ffhq -> 0,
else -> 1), so a run must pool at least one ffhq* dataset and one generator
dataset — see data.py.

Needs pandas + numpy always, scikit-learn + joblib only to actually fit.
"""

import argparse
import os
import sys

import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FOLDER)
for path in (FOLDER, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import data as data_mod  # noqa: E402
import evaluation  # noqa: E402
from metrics import THRESHOLD  # noqa: E402
from model import BACKENDS, GradientBoostingEnsemble  # noqa: E402


def report_pooled(pooled, train_mask, val_mask, args):
    print("=" * 78)
    print("POOLED DESIGN MATRIX")
    print("=" * 78)
    print(f"datasets   : {', '.join(pooled.datasets)}")
    print(f"models     : {', '.join(pooled.stems)}  (features, in order)")
    print(f"transform  : {pooled.transform}     resize regime: {pooled.resize}")
    print(f"shape      : X={pooled.X.shape}  y={pooled.y.shape}")
    print()
    print(pooled.summary().to_string(index=False))
    print()

    n_pos = int((pooled.y == 1).sum())
    n_neg = int((pooled.y == 0).sum())
    print(f"class balance : fake(1)={n_pos}  real(0)={n_neg}  "
          f"({n_pos / max(n_pos + n_neg, 1):.1%} fake)")
    if min(n_pos, n_neg) and max(n_pos, n_neg) / min(n_pos, n_neg) > 3:
        print("  note: >3:1 imbalance. --class-weight balanced (the default) "
              "compensates; with 'none' the booster can score well by answering "
              "with the majority class.")
    print()

    if args.split == "by_dataset":
        desc = f"by_dataset (holdout: {', '.join(args.holdout_dataset or [])})"
    elif args.split == "by_shard":
        held = sorted(set(data_mod.shard_groups(pooled.X.index)[val_mask.to_numpy()]))
        desc = (f"by_shard (val_frac={args.val_frac}, seed={args.seed}; "
                f"{len(held)} shard(s) held out: {', '.join(held)})")
    else:
        desc = f"random stratified (val_frac={args.val_frac}, seed={args.seed})"
    print(f"split      : {desc}")
    print(f"             train={int(train_mask.sum())} rows, val={int(val_mask.sum())} rows")
    for warn in data_mod.split_warnings(pooled, val_mask):
        print(f"  WARNING: {warn}")
    if args.split == "random":
        print("  note: a random split puts images from the same shard on both sides, "
              "so val measures fit rather than generalization. --split by_shard holds "
              "out whole shards; --split by_dataset holds out whole generators.")
    print()

    # Per-detector baselines, so any booster result has a like-for-like
    # reference (ensemble.py prints the same idea per dataset).
    print(f"per-detector baseline on the pooled set (raw CSV scores, threshold {THRESHOLD}):")
    raw = data_mod.build_pooled(pooled.datasets, pooled.stems, transform="raw")
    truth = raw.y.to_numpy()
    print(f"  {'model':32} {'accuracy':>10} {'fake-rate':>10}")
    for stem in pooled.stems:
        pred = (raw.X[stem].to_numpy() >= THRESHOLD).astype(int)
        print(f"  {stem:32} {float((pred == truth).mean()):>10.4f} {float(pred.mean()):>10.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", action="append",
                        help="dataset to pool (repeatable). Default: every dataset with CSVs. "
                             "Must include at least one ffhq* (real) and one generator (fake).")
    parser.add_argument("--models",
                        help=f"comma-separated detector stems to use as features "
                             f"({data_mod.MIN_MODELS}-{data_mod.MAX_MODELS}).")
    parser.add_argument("--transform", default="raw", choices=data_mod.TRANSFORMS,
                        help="per-model feature transform (default: raw; trees are "
                             "invariant to monotone rescaling, so this barely matters here).")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"fake/real cutoff on the ensemble score (default: {THRESHOLD}).")
    parser.add_argument("--class-weight", default="balanced", choices=data_mod.CLASS_WEIGHTS,
                        help="compensate for the real/fake imbalance (default: balanced).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name",
                        help="artifact directory name under results/gradient_boosting/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="build features, print the plan and configuration, fit nothing.")
    parser.add_argument("--list", action="store_true",
                        help="list datasets and the stems selectable across them, then exit.")

    split = parser.add_argument_group("split")
    split.add_argument("--split", default="random", choices=data_mod.SPLITS)
    split.add_argument("--val-frac", type=float, default=0.2)
    split.add_argument("--holdout-dataset", action="append",
                       help="with --split by_dataset: hold this dataset out entirely "
                            "(repeatable).")

    gb = parser.add_argument_group("booster")
    gb.add_argument("--backend", default="sklearn_gb", choices=BACKENDS)
    gb.add_argument("--n-estimators", type=int, default=300)
    gb.add_argument("--learning-rate", type=float, default=0.05)
    gb.add_argument("--max-depth", type=int, default=3)
    gb.add_argument("--subsample", type=float, default=0.8)
    gb.add_argument("--min-samples-leaf", type=int, default=20)
    gb.add_argument("--max-leaf-nodes", type=int, default=31)
    gb.add_argument("--l2", type=float, default=0.0)
    gb.add_argument("--validation-fraction", type=float, default=0.1)
    gb.add_argument("--n-iter-no-change", type=int, default=20)

    args = parser.parse_args()

    datasets = args.dataset or data_mod.discover_datasets()
    if not datasets:
        raise SystemExit(
            "no prediction CSVs found in results/. Run run.py first (see CLAUDE.md)."
        )

    if args.list:
        print("datasets with result CSVs:")
        for ds in datasets:
            stems = data_mod.model_stems_for(ds)
            print(f"  {ds:28} {len(stems):>3} model(s): {', '.join(stems) or '(none)'}")
        shared = data_mod.common_stems(datasets)
        print(f"\nstems present for ALL of the above ({len(shared)}): "
              f"{', '.join(shared) or '(none)'}")
        print(f"pick {data_mod.MIN_MODELS}-{data_mod.MAX_MODELS} of those for --models.")
        return

    if not args.models:
        shared = data_mod.common_stems(datasets)
        raise SystemExit(
            f"--models is required ({data_mod.MIN_MODELS}-{data_mod.MAX_MODELS} stems). "
            f"Available across the selected datasets: {', '.join(shared) or '(none)'}. "
            "Run with --list for the full picture."
        )
    stems = [s.strip() for s in args.models.split(",") if s.strip()]
    data_mod.validate_selection(stems)

    gaps = {
        ds: [s for s in stems if s not in data_mod.model_stems_for(ds)] for ds in datasets
    }
    gaps = {ds: missing for ds, missing in gaps.items() if missing}
    if gaps:
        detail = "; ".join(f"{ds} lacks {missing}" for ds, missing in gaps.items())
        raise SystemExit(
            f"every selected model needs a CSV for every pooled dataset ({detail}). "
            "Either run those combinations or drop the dataset/model."
        )

    pooled = data_mod.build_pooled(datasets, stems, transform=args.transform)
    train_mask, val_mask = data_mod.make_split(
        pooled, mode=args.split, val_frac=args.val_frac, seed=args.seed,
        holdout_datasets=args.holdout_dataset,
    )
    report_pooled(pooled, train_mask, val_mask, args)

    est = GradientBoostingEnsemble(
        n_features=pooled.n_features,
        backend=args.backend,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        random_state=args.seed,
    )
    print("-" * 78)
    print(est.describe())
    print(f"  class weight      : {args.class_weight}")
    print("-" * 78)
    print()

    if args.dry_run:
        print("[dry-run] setup verified; nothing fitted, nothing written.\n")
        return

    tr, va = train_mask.to_numpy(), val_mask.to_numpy()
    X_train, y_train = pooled.X[tr], pooled.y[tr]
    X_val, y_val = pooled.X[va], pooled.y[va]
    weights = data_mod.sample_weights(y_train.to_numpy(), args.class_weight)

    print(f"fitting on {len(X_train)} rows ...")
    est.fit(X_train, y_train, sample_weight=weights)
    used = est.n_trees_used()
    if used is not None:
        print(f"  {used} boosting round(s) used of {args.n_estimators} allowed")

    train_metrics = evaluation.evaluate(
        y_train, est.predict_proba(X_train), pooled.groups[tr], args.threshold
    )
    val_metrics = evaluation.evaluate(
        y_val, est.predict_proba(X_val), pooled.groups[va], args.threshold
    )
    print()
    evaluation.print_metrics("train", train_metrics)
    print()
    evaluation.print_metrics("validation", val_metrics)

    importance = est.feature_importance()
    if importance:
        print("\nfeature importance:")
        for stem, value in sorted(importance.items(), key=lambda kv: -kv[1]):
            print(f"  {stem:32} {value:>8.4f}")

    out_dir = evaluation.run_dir(args.run_name or f"gb_{pooled.n_features}model")
    val_pred = pd.DataFrame(
        {"score": est.predict_proba(X_val), "label": y_val.to_numpy()}, index=X_val.index
    )
    val_pred["Predicted label"] = (val_pred["score"] >= args.threshold).astype(int)
    evaluation.write_artifacts(
        out_dir,
        {
            "method": "gradient_boosting",
            "stems": pooled.stems,
            "transform": pooled.transform,
            "resize": pooled.resize,
            "datasets": pooled.datasets,
            "threshold": args.threshold,
            "class_weight": args.class_weight,
            "split": {
                "mode": args.split,
                "val_frac": args.val_frac,
                "seed": args.seed,
                "holdout_datasets": args.holdout_dataset or [],
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
            },
            "hyperparams": est.hyperparams(),
            "n_trees_used": used,
        },
        metrics={"train": train_metrics, "validation": val_metrics},
        val_predictions=val_pred,
    )
    est.save(out_dir)
    print(f"\nwrote artifacts to {os.path.relpath(out_dir, REPO_ROOT)}/\n")


if __name__ == "__main__":
    main()
