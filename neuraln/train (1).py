#!/usr/bin/env python3
"""Train the neural-network ensemble over the per-detector results CSVs.

    # what can I select right now?
    python3 neural_network/train.py --list

    # verify the setup without training (needs no torch)
    python3 neural_network/train.py \
        --models freqnet_GAN,rine1_DM,grag2021_latent_DM \
        --dataset ffhq_test --dataset flux_test --dry-run

    # train, once you're ready
    python3 neural_network/train.py \
        --models freqnet_GAN,rine1_DM,grag2021_latent_DM,cnndetection \
        --dataset ffhq_test --dataset flux_test --run-name nn_4model

--models takes 3-5 detector stems; each becomes one input node, in the order
given. That order is part of the trained model — the frozen per-column scaler in
the checkpoint is aligned to it — and is recorded in the run's config.json.

TRAINING IS NEVER IMPLICIT. It happens only when --dry-run is absent, and
--dry-run is what to use while wiring this up: it builds the real design matrix,
reports class balance and the split, prints per-detector baselines and the exact
architecture (layer sizes, parameter count, loss, optimizer), then stops.

Ground truth comes from the dataset name (loaders.label_for_dataset: ffhq -> 0,
else -> 1), so a run must pool at least one ffhq* dataset and one generator
dataset — see data.py.

Needs pandas + numpy always, torch only to actually train.
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
from model import ACTIVATIONS, MLPEnsemble  # noqa: E402


def report_pooled(pooled, train_mask, val_mask, args):
    print("=" * 78)
    print("POOLED DESIGN MATRIX")
    print("=" * 78)
    print(f"datasets   : {', '.join(pooled.datasets)}")
    print(f"models     : {', '.join(pooled.stems)}  (input nodes, in order)")
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
        print("  note: >3:1 imbalance. --class-weight balanced (the default) sets "
              "the loss's pos_weight; with 'none' the network can drive its output "
              "bias to the majority class and stall there.")
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
    print("             (validation is also the early-stopping signal)")
    for warn in data_mod.split_warnings(pooled, val_mask):
        print(f"  WARNING: {warn}")
    if args.split == "random":
        print("  note: a random split puts images from the same shard on both sides, "
              "so val measures fit rather than generalization. --split by_shard holds "
              "out whole shards; --split by_dataset holds out whole generators.")
    print()

    # Per-detector baselines, so any trained result has a like-for-like
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
                        help=f"comma-separated detector stems, one input node each "
                             f"({data_mod.MIN_MODELS}-{data_mod.MAX_MODELS}).")
    parser.add_argument("--transform", default="logit", choices=data_mod.TRANSFORMS,
                        help="per-model feature transform (default: logit; raw "
                             "probabilities saturate and are hard for a linear layer).")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"fake/real cutoff on the sigmoid output (default: {THRESHOLD}).")
    parser.add_argument("--class-weight", default="balanced", choices=data_mod.CLASS_WEIGHTS,
                        help="sets the loss's pos_weight (default: balanced).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name",
                        help="artifact directory name under results/neural_network/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="build features, print the plan and architecture, train nothing.")
    parser.add_argument("--list", action="store_true",
                        help="list datasets and the stems selectable across them, then exit.")

    split = parser.add_argument_group("split")
    split.add_argument("--split", default="random", choices=data_mod.SPLITS)
    split.add_argument("--val-frac", type=float, default=0.2)
    split.add_argument("--holdout-dataset", action="append",
                       help="with --split by_dataset: hold this dataset out entirely "
                            "(repeatable).")

    nn = parser.add_argument_group("network")
    nn.add_argument("--hidden", default="16,8",
                    help="comma-separated hidden widths; '' = no hidden layer, i.e. "
                         "logistic regression on the scores (default: 16,8).")
    nn.add_argument("--activation", default="relu", choices=ACTIVATIONS)
    nn.add_argument("--dropout", type=float, default=0.0)
    nn.add_argument("--epochs", type=int, default=200)
    nn.add_argument("--batch-size", type=int, default=256)
    nn.add_argument("--learning-rate", type=float, default=1e-3)
    nn.add_argument("--weight-decay", type=float, default=1e-4)
    nn.add_argument("--patience", type=int, default=20,
                    help="stop after this many epochs without validation improvement "
                         "(0 disables early stopping).")
    nn.add_argument("--device", default="auto", help="auto | cpu | cuda")

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

    tr, va = train_mask.to_numpy(), val_mask.to_numpy()
    X_train, y_train = pooled.X[tr], pooled.y[tr]
    X_val, y_val = pooled.X[va], pooled.y[va]

    net = MLPEnsemble(
        n_features=pooled.n_features,
        hidden=args.hidden,
        activation=args.activation,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        pos_weight=data_mod.pos_weight(y_train.to_numpy(), args.class_weight),
        seed=args.seed,
        device=args.device,
    )
    print("-" * 78)
    print(net.describe())
    print(f"  class weight      : {args.class_weight}")
    print("-" * 78)
    print()

    if args.dry_run:
        print("[dry-run] setup verified; nothing trained, nothing written.\n")
        return

    print(f"training on {len(X_train)} rows, validating on {len(X_val)} ...")
    net.fit(X_train, y_train, eval_set=(X_val.to_numpy(), y_val.to_numpy()))
    print(f"  ran {len(net.history)} epoch(s); best was epoch {net.best_epoch} "
          f"(val loss {net.best_loss:.5f}), weights restored to it")

    train_metrics = evaluation.evaluate(
        y_train, net.predict_proba(X_train), pooled.groups[tr], args.threshold
    )
    val_metrics = evaluation.evaluate(
        y_val, net.predict_proba(X_val), pooled.groups[va], args.threshold
    )
    print()
    evaluation.print_metrics("train", train_metrics)
    print()
    evaluation.print_metrics("validation", val_metrics)

    out_dir = evaluation.run_dir(args.run_name or f"nn_{pooled.n_features}model")
    val_pred = pd.DataFrame(
        {"score": net.predict_proba(X_val), "label": y_val.to_numpy()}, index=X_val.index
    )
    val_pred["Predicted label"] = (val_pred["score"] >= args.threshold).astype(int)
    evaluation.write_artifacts(
        out_dir,
        {
            "method": "neural_network",
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
            "hyperparams": net.hyperparams(),
            "best_epoch": net.best_epoch,
            "epochs_run": len(net.history),
        },
        metrics={"train": train_metrics, "validation": val_metrics},
        val_predictions=val_pred,
    )
    net.save(out_dir)
    net.write_history(out_dir)
    print(f"\nwrote artifacts to {os.path.relpath(out_dir, REPO_ROOT)}/\n")


if __name__ == "__main__":
    main()
