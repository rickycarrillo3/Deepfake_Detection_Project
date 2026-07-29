"""The gradient-boosting ensemble: 3-5 detector scores -> fake-probability.

One row is one image, one column is one detector's score for it, and the target
is the image's ground-truth label. The booster fits shallow trees on those
columns in stages, each correcting the previous stage's residual error.

What this buys over ensemble.py's untrained rules. mean and logodds apply one
fixed formula to every image; hard_or fires if any single model does. A booster
learns conditional structure instead — "trust freqnet only when grag2021 also
fires", "rine's score only matters above 0.9", "when these two disagree, side
with the third". Those are axis-aligned interactions the fixed rules cannot
express, and they are precisely what you would hope to find given the
low-correlation model selection that failure_correlation.py exists to do.

Why trees suit this particular table. The features are few (3-5), on wildly
different and non-comparable scales, and each is roughly monotone in the label.
Trees split on thresholds, so they are invariant to any monotone rescaling of a
column — the per-model calibration problem that forces ensemble.py to offer
--norm simply does not arise. That is also why --transform barely moves the
result here, unlike for the neural network.

Backends
--------
  sklearn_gb    GradientBoostingClassifier — the classical stage-wise booster.
                Exact, deterministic given a seed, fine to roughly 1e5 rows.
  sklearn_hist  HistGradientBoostingClassifier — bins features into histograms,
                much faster past ~1e5 rows (ten shards across several datasets
                gets there) with built-in early stopping.

Both emit calibrated-ish probabilities through predict_proba, so
metrics.THRESHOLD keeps its meaning downstream — the same reason ensemble.py's
`raw` norm is its evaluatable default.

Not used: xgboost / lightgbm. They would be a new heavyweight dependency for a
five-column table when scikit-learn is already in the project (aide and co-spy
both pull it in). Swappable later behind the same `backend` switch if the
histogram backend ever proves too slow.
"""

import os

import numpy as np

BACKENDS = ("sklearn_gb", "sklearn_hist")
MODEL_FILE = "model.joblib"


class GradientBoostingEnsemble:
    """Fit on a pooled score matrix; predicts P(fake) per image.

      est = GradientBoostingEnsemble(n_features=3)
      est.fit(X_train, y_train, sample_weight=w)
      probabilities = est.predict_proba(X_val)
    """

    def __init__(
        self,
        n_features,
        backend="sklearn_gb",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=20,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=0,
    ):
        if backend not in BACKENDS:
            raise SystemExit(
                f"unknown backend '{backend}' (choose from {', '.join(BACKENDS)})"
            )
        self.n_features = int(n_features)
        self.backend = backend
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.l2_regularization = float(l2_regularization)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.random_state = int(random_state)
        self.clf = None
        self.stems = None      # feature names, for readable importances

    # -- introspection ------------------------------------------------------

    def hyperparams(self):
        """JSON-serializable, recorded into the run's config.json."""
        common = {
            "backend": self.backend,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_iter_no_change": self.n_iter_no_change,
            "validation_fraction": self.validation_fraction,
            "min_samples_leaf": self.min_samples_leaf,
            "max_depth": self.max_depth,
        }
        if self.backend == "sklearn_gb":
            common.update({"n_estimators": self.n_estimators, "subsample": self.subsample})
        else:
            common.update({
                "max_iter": self.n_estimators,
                "max_leaf_nodes": self.max_leaf_nodes,
                "l2_regularization": self.l2_regularization,
            })
        return common

    def describe(self):
        """Printable without scikit-learn installed and without fitting, so
        --dry-run works in any interpreter."""
        lines = [
            f"gradient boosting ({self.backend})",
            f"  inputs            : {self.n_features} detector scores",
            f"  boosting rounds   : {self.n_estimators} (early stop after "
            f"{self.n_iter_no_change} rounds without improvement)",
            f"  learning rate     : {self.learning_rate}",
            f"  tree depth        : {self.max_depth}",
        ]
        if self.backend == "sklearn_gb":
            lines.append(f"  row subsample     : {self.subsample}")
        else:
            lines.append(f"  max leaf nodes    : {self.max_leaf_nodes}")
            lines.append(f"  l2 regularization : {self.l2_regularization}")
        lines.append(f"  min samples/leaf  : {self.min_samples_leaf}")
        lines.append("  output            : predict_proba[:, 1] = P(fake)")
        return "\n".join(lines)

    # -- fit / predict ------------------------------------------------------

    def _build(self):
        if self.backend == "sklearn_gb":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                min_samples_leaf=self.min_samples_leaf,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state,
            )
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )

    def fit(self, X, y, sample_weight=None):
        """Both backends carve their own early-stopping slice out of these rows
        via validation_fraction, so the held-out validation set stays untouched."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            raise SystemExit(
                "scikit-learn is required to train the gradient-boosting ensemble.\n"
                "  gradient_boosting/setup.sh builds a venv with it, or use a model "
                "venv that already has it:\n"
                "    models/aide/venv/bin/python gradient_boosting/train.py ...\n"
                "(--dry-run works without it.)"
            )
        self.stems = list(getattr(X, "columns", [])) or None
        self.clf = self._build()
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int),
                     sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        if self.clf is None:
            raise SystemExit("ensemble is not fitted yet.")
        return self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self):
        """Per-detector importance, named. None when the backend exposes none —
        the histogram backend does not; use sklearn.inspection.permutation_importance
        there if you need it."""
        if self.clf is None or not hasattr(self.clf, "feature_importances_"):
            return None
        names = self.stems or [f"f{i}" for i in range(self.n_features)]
        return dict(zip(names, (float(v) for v in self.clf.feature_importances_)))

    def n_trees_used(self):
        """Rounds actually fit, which early stopping may cut below n_estimators."""
        if self.clf is None:
            return None
        for attr in ("n_estimators_", "n_iter_"):
            if hasattr(self.clf, attr):
                return int(getattr(self.clf, attr))
        return None

    # -- persistence --------------------------------------------------------

    def save(self, out_dir):
        import joblib

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, MODEL_FILE)
        joblib.dump(
            {"clf": self.clf, "stems": self.stems, "hyperparams": self.hyperparams()}, path
        )
        return path

    @classmethod
    def load(cls, out_dir):
        import joblib

        blob = joblib.load(os.path.join(out_dir, MODEL_FILE))
        stems = blob.get("stems") or []
        obj = cls(n_features=len(stems))
        obj.clf = blob["clf"]
        obj.stems = stems or None
        return obj
