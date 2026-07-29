"""The neural-network ensemble: 3-5 detector scores -> one sigmoid fake-probability.

The shape:

    image ──> detector 1 ─┐
          ──> detector 2 ─┤
          ──> detector 3 ─┼──> [ MLP ] ──> sigmoid ──> P(fake) ──> >= 0.5 ? fake
          ──> detector 4 ─┤
          ──> detector 5 ─┘
        (run.py already did this part,        this module
         the scores are the results CSVs)

The image never reaches this module. run.py has already scored it with each
detector and written the score to results/<dataset>_<stem>_predictions.csv, so
the input layer is 3-5 numbers, one per selected detector, in a fixed column
order. That is what makes this trainable on a laptop: the expensive part is
cached on disk and this is a few hundred parameters on top. It also means the
input width is set at construction from how many models you select — three
stems build three input nodes, five build five.

The output head
---------------
One linear unit to a single logit, trained with BCEWithLogitsLoss. That IS the
sigmoid: BCEWithLogits applies it internally in a log-sum-exp-stable way rather
than doing sigmoid-then-log in two steps, which overflows once a confident logit
passes roughly +/-30. Hence no nn.Sigmoid in the Sequential — predict_proba()
applies torch.sigmoid explicitly at inference, and what comes out is a
probability in (0, 1) thresholded at metrics.THRESHOLD exactly like every other
score in this project.

Standardization
---------------
A network is very much not invariant to input scale (a tree would be). Raw
detector probabilities pile up on spikes at 0 and 1; even after data.py's logit
transform the columns sit at different centers and spreads, and an unscaled
input layer spends its early epochs just undoing that. So the mean and standard
deviation are computed on the TRAINING rows only and frozen into the checkpoint,
and validation and later inference are transformed by those training statistics
— not their own, which would leak and would be uncomputable for a single image.

Size
----
Default hidden layers are (16, 8) — deliberately tiny. With 3-5 inputs there is
little to learn beyond a smooth non-linear surface, and this is a meta-learner
sitting on a handful of single-class shards; a wide net memorizes which dataset
a score pattern came from. An empty --hidden gives a plain logistic regression
on the scores, which is a genuinely useful baseline: if the MLP cannot beat it,
the extra capacity is not buying anything.

torch is imported lazily inside the methods that need it, so describe() and
train.py's --dry-run work in any interpreter — the same reason ensemble.py
sticks to pandas + numpy.
"""

import json
import os

import numpy as np

MODEL_FILE = "model.pt"
ACTIVATIONS = ("relu", "gelu", "tanh")


def parse_hidden(spec):
    """'16,8' -> (16, 8). Empty -> () = logistic regression on the scores."""
    if isinstance(spec, (list, tuple)):
        return tuple(int(v) for v in spec)
    spec = (spec or "").strip()
    if not spec:
        return ()
    try:
        sizes = tuple(int(part) for part in spec.split(",") if part.strip())
    except ValueError:
        raise SystemExit(f"--hidden must be comma-separated integers, got '{spec}'")
    if any(s <= 0 for s in sizes):
        raise SystemExit(f"--hidden sizes must be positive, got '{spec}'")
    return sizes


class MLPEnsemble:
    """Fit on a pooled score matrix; predicts P(fake) per image.

      net = MLPEnsemble(n_features=3, hidden=(16, 8))
      net.fit(X_train, y_train, eval_set=(X_val, y_val))
      probabilities = net.predict_proba(X_val)
    """

    def __init__(
        self,
        n_features,
        hidden=(16, 8),
        activation="relu",
        dropout=0.0,
        epochs=200,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=20,
        pos_weight=1.0,
        seed=0,
        device="auto",
    ):
        if activation not in ACTIVATIONS:
            raise SystemExit(
                f"unknown activation '{activation}' (choose from {', '.join(ACTIVATIONS)})"
            )
        if not 0.0 <= dropout < 1.0:
            raise SystemExit(f"dropout must be in [0, 1), got {dropout}")
        self.n_features = int(n_features)
        self.hidden = parse_hidden(hidden)
        self.activation = activation
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.pos_weight = float(pos_weight)
        self.seed = int(seed)
        self.device = device
        self.net = None
        self.mean_ = None      # training-set standardization, frozen at fit time
        self.std_ = None
        self.stems = None
        self.history = []
        self.best_epoch = None
        self.best_loss = None

    # -- introspection ------------------------------------------------------

    def layer_sizes(self):
        return (self.n_features,) + tuple(self.hidden) + (1,)

    def n_parameters(self):
        sizes = self.layer_sizes()
        return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))

    def hyperparams(self):
        """JSON-serializable, recorded into the run's config.json."""
        return {
            "hidden": list(self.hidden),
            "activation": self.activation,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "pos_weight": self.pos_weight,
            "seed": self.seed,
            "n_parameters": self.n_parameters(),
        }

    def describe(self):
        """Printable without torch installed and without fitting, so --dry-run
        works in any interpreter."""
        sizes = self.layer_sizes()
        arrow = "  ->  ".join(
            f"[{n}]" + ("" if i in (0, len(sizes) - 1) else f" {self.activation}")
            for i, n in enumerate(sizes)
        )
        return "\n".join([
            "neural network (MLP)",
            f"  inputs            : {self.n_features} detector scores (one input node each)",
            f"  architecture      : {arrow}",
            "  output            : 1 logit -> sigmoid -> P(fake)",
            f"  loss              : BCEWithLogitsLoss (pos_weight={self.pos_weight:.4f})",
            f"  trainable params  : {self.n_parameters()}",
            f"  optimizer         : Adam(lr={self.learning_rate}, "
            f"weight_decay={self.weight_decay})",
            f"  dropout           : {self.dropout}",
            f"  epochs            : {self.epochs} (early stop patience {self.patience})",
            f"  batch size        : {self.batch_size}",
            "  input scaling     : z-score, statistics frozen from the training rows",
        ])

    # -- torch plumbing -----------------------------------------------------

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError:
            raise SystemExit(
                "PyTorch is required to train or run the neural-network ensemble.\n"
                "  neural_network/setup.sh builds a venv with it, or use any model "
                "venv that already has torch:\n"
                "    models/freqnet/venv/bin/python neural_network/train.py ...\n"
                "(--dry-run works without torch.)"
            )
        return torch

    def _resolve_device(self):
        torch = self._torch()
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self):
        torch = self._torch()
        from torch import nn

        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[self.activation]
        torch.manual_seed(self.seed)

        layers = []
        prev = self.n_features
        for width in self.hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(act())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = width
        # Single logit. No Sigmoid module here on purpose: the sigmoid lives in
        # BCEWithLogitsLoss during training and in predict_proba at inference.
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    # -- standardization ----------------------------------------------------

    def _fit_scaler(self, X):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0      # a constant column carries no information; don't divide by 0
        self.std_ = std

    def _scale(self, X):
        if self.mean_ is None:
            raise SystemExit("scaler not fitted; call fit() first.")
        return (X - self.mean_) / self.std_

    # -- fit / predict ------------------------------------------------------

    def fit(self, X, y, eval_set=None):
        """Train, early-stopping on `eval_set` (X_val, y_val) when given.

        The best weights by validation loss are restored at the end, not the
        last epoch's — with a small net on noisy meta-features the final epoch is
        frequently worse than something 20 epochs back.
        """
        torch = self._torch()
        from torch import nn

        self.stems = list(getattr(X, "columns", [])) or None
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self._fit_scaler(X)
        device = self._resolve_device()
        Xt = torch.tensor(self._scale(X), dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)

        has_val = eval_set is not None
        if has_val:
            Xv_raw, yv_raw = eval_set
            Xv = torch.tensor(
                self._scale(np.asarray(Xv_raw, dtype=np.float32)),
                dtype=torch.float32, device=device,
            )
            yv = torch.tensor(
                np.asarray(yv_raw, dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32, device=device,
            )

        self.net = self._build().to(device)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], dtype=torch.float32, device=device)
        )

        n = len(Xt)
        generator = torch.Generator().manual_seed(self.seed)
        best = {"loss": float("inf"), "epoch": -1, "state": None}
        self.history = []

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=generator).to(device)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss = criterion(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            train_loss = total / n

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_loss = float(criterion(self.net(Xv), yv))
            else:
                val_loss = train_loss

            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            if val_loss < best["loss"] - 1e-6:
                best = {
                    "loss": val_loss,
                    "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in self.net.state_dict().items()},
                }
            elif self.patience and epoch - best["epoch"] >= self.patience:
                break

        if best["state"] is not None:
            self.net.load_state_dict(best["state"])
        self.best_epoch = best["epoch"]
        self.best_loss = best["loss"]
        return self

    def predict_proba(self, X):
        """P(fake) per row — the sigmoid of the output logit."""
        torch = self._torch()
        if self.net is None:
            raise SystemExit("network is not fitted yet.")
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                torch.tensor(
                    self._scale(np.asarray(X, dtype=np.float32)),
                    dtype=torch.float32, device=device,
                )
            )
            return torch.sigmoid(logits).cpu().numpy().reshape(-1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- persistence --------------------------------------------------------

    def save(self, out_dir):
        torch = self._torch()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, MODEL_FILE)
        torch.save(
            {
                "state_dict": None if self.net is None else self.net.state_dict(),
                "n_features": self.n_features,
                "hyperparams": self.hyperparams(),
                "stems": self.stems,
                # The scaler travels with the weights: without these exact
                # numbers the checkpoint produces garbage on new data.
                "mean": None if self.mean_ is None else self.mean_.tolist(),
                "std": None if self.std_ is None else self.std_.tolist(),
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, out_dir, map_location="cpu"):
        torch = cls._torch()
        path = os.path.join(out_dir, MODEL_FILE)
        # weights_only landed in torch 1.13 and its default flipped to True in
        # 2.6. This checkpoint is our own and holds plain lists/dicts alongside
        # the tensors, so ask for the permissive read and fall back on older
        # torches that do not know the argument.
        try:
            blob = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location=map_location)

        hp = blob["hyperparams"]
        obj = cls(
            n_features=blob["n_features"],
            hidden=tuple(hp["hidden"]),
            activation=hp["activation"],
            dropout=hp["dropout"],
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            patience=hp["patience"],
            pos_weight=hp["pos_weight"],
            seed=hp["seed"],
        )
        obj.net = obj._build()
        if blob["state_dict"] is not None:
            obj.net.load_state_dict(blob["state_dict"])
        obj.mean_ = None if blob["mean"] is None else np.asarray(blob["mean"], dtype=np.float32)
        obj.std_ = None if blob["std"] is None else np.asarray(blob["std"], dtype=np.float32)
        obj.stems = blob.get("stems")
        obj.history = blob.get("history", [])
        return obj

    def write_history(self, out_dir):
        """Per-epoch loss curves next to the checkpoint — the first thing to look
        at before touching any hyperparameter."""
        if not self.history:
            return None
        path = os.path.join(out_dir, "training_history.json")
        with open(path, "w") as fh:
            json.dump(self.history, fh, indent=2)
        return path
