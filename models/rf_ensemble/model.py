"""
Random-forest ensemble meta-model -- implements the models/<model_name>/
contract:
    load()               -> load the trained RandomForestClassifier
    predict(image_path)  -> float in [0, 1]

Does NOT run any of the underlying (expert) models' own code. It looks up
each expert's already-computed score for this exact image from
results/<dataset>_<stem>_predictions.csv (written by run.py when you scored
that model on that dataset), assembles the score vector, and feeds it to the
trained random forest. This only works for images that have already been
scored by every expert the model was trained with -- it is a fusion-over-
existing-results model, not a standalone detector for arbitrary new images.
Same approach as models/gated_ensemble/model.py, minus the image-pixel input
(a random forest here trains on expert scores alone, not pixels).
"""
import os
import re

import joblib
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(MODEL_DIR))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, "rf_ensemble.joblib")

_DATA_PATH_RE = re.compile(r"/data/(?P<dataset>[^/]+)/(?P<rel>.+)$")


def _dataset_and_relpath(image_path):
    """Recover (dataset_name, filepath-as-written-in-the-CSVs) from an
    absolute image path under data/<dataset_name>/... -- matching run.py's
    own loaders.dataset layout assumption. Same helper as
    models/gated_ensemble/model.py."""
    m = _DATA_PATH_RE.search(image_path.replace(os.sep, "/"))
    if not m:
        raise ValueError(
            f"Could not determine dataset from path {image_path!r} -- expected "
            f".../data/<dataset_name>/... Adjust _dataset_and_relpath if your "
            f"layout differs."
        )
    return m.group("dataset"), m.group("rel")


class RFEnsembleModel:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or os.environ.get("RF_ENSEMBLE_CHECKPOINT", DEFAULT_CHECKPOINT)
        self.clf = None
        self.stems = None
        self._score_tables = {}  # dataset_name -> {stem: DataFrame}

    def load(self):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Missing trained checkpoint at {self.checkpoint_path}. Train one "
                f"with train.py first, e.g.:\n"
                f"  models/rf_ensemble/venv/bin/python train.py --datasets <dataset...> "
                f"--stems <stem...> --out {self.checkpoint_path}\n"
                f"or set RF_ENSEMBLE_CHECKPOINT=/path/to/it.joblib"
            )
        saved = joblib.load(self.checkpoint_path)
        self.clf = saved["model"]
        self.stems = saved["stems"]
        return self

    def _scores_for(self, dataset_name, relpath):
        if dataset_name not in self._score_tables:
            tables = {}
            for stem in self.stems:
                path = os.path.join(RESULTS_DIR, f"{dataset_name}_{stem}_predictions.csv")
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"Missing {path} -- score this dataset with the model that "
                        f"produces stem '{stem}' first."
                    )
                tables[stem] = pd.read_csv(path).set_index("filepath")
            self._score_tables[dataset_name] = tables

        tables = self._score_tables[dataset_name]
        scores = []
        for stem in self.stems:
            if relpath not in tables[stem].index:
                raise KeyError(f"{relpath!r} has no precomputed score in stem '{stem}' for dataset '{dataset_name}'.")
            scores.append(float(tables[stem].loc[relpath, "score"]))
        return scores

    def predict(self, image_path):
        if self.clf is None:
            raise RuntimeError("Call load() before predict().")

        dataset_name, relpath = _dataset_and_relpath(image_path)
        scores = self._scores_for(dataset_name, relpath)

        prob_fake = self.clf.predict_proba([scores])[0, 1]
        return float(prob_fake)


_DEFAULT_MODEL = None


def load(**kwargs):
    """Module-level contract: load the trained random forest and return a
    ready RFEnsembleModel. checkpoint_path= optionally overrides which
    trained .joblib to use, e.g. --model-arg checkpoint_path=/path/to/other.joblib"""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = RFEnsembleModel(**kwargs).load()
    return _DEFAULT_MODEL


def predict(image_path):
    """Module-level contract: predict fake-probability for one image path."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)


def result_stem():
    """Optional runner contract (see run.py): distinguishes results from
    different trained forests (different expert sets) if you ever train more
    than one, by naming the CSV after the checkpoint file used."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before result_stem().")
    checkpoint_name = os.path.splitext(os.path.basename(_DEFAULT_MODEL.checkpoint_path))[0]
    return f"rf_ensemble_{checkpoint_name}" if checkpoint_name != "rf_ensemble" else "rf_ensemble"
