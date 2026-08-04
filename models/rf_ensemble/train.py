"""
Trains a RandomForestClassifier on already-computed per-model scores (see
dataset.py -- this does not call any expert model's own code, only reads
results/*.csv). You pick which 3-5 expert models to use via --stems.

Usage:
    models/rf_ensemble/venv/bin/python train.py \
        --datasets ffhq_faces flux_faces stylegan-0.5_faces stylegan-0.7_faces \
        --stems aide_DM co-spy_GAN cnndetection grag2021_latent \
        --out models/rf_ensemble/rf_ensemble.joblib
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from dataset import build_training_table


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", required=True, help="dataset names under data/ to train on")
    p.add_argument("--stems", nargs="+", required=True,
                    help="3-5 expert model CSV stems to use as features, e.g. aide_DM co-spy_GAN cnndetection")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_ensemble.joblib"))
    args = p.parse_args()

    X, y, keys = build_training_table(args.datasets, args.stems)
    print(f"Loaded {len(y)} example(s): {int(y.sum())} fake, {int((1 - y).sum())} real")

    X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
        X, y, keys, test_size=args.test_fraction, random_state=args.seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    print(f"\nTest accuracy: {accuracy_score(y_test, test_preds):.2%}")
    try:
        print(f"Test ROC AUC:  {roc_auc_score(y_test, test_probs):.4f}")
    except ValueError as e:
        print(f"Test ROC AUC:  n/a ({e})")
    print("\n" + classification_report(y_test, test_preds, target_names=["real", "fake"]))

    print("\nFeature importances (per expert):")
    for stem, importance in sorted(zip(args.stems, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {stem:30} {importance:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump({"model": clf, "stems": args.stems}, args.out)
    print(f"\nSaved trained model + stems to {args.out}")


if __name__ == "__main__":
    main()
